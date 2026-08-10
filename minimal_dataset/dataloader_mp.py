"""
Multi-processing DataLoader with staging queue and batch queue.
Generic version — works with any dataset, not just Parquet.
Each worker creates its own dataset instance from a factory function.
"""
import multiprocessing as mp
import time
import json
import os

import torch

from .sampler import LockFreeSampler


def _default_collate_fn(samples):
    images = torch.stack([s[0] for s in samples])
    labels = [s[1] for s in samples]
    if isinstance(labels[0], str):
        labels = torch.tensor([int(l.replace('n', '')) for l in labels])
    else:
        labels = torch.tensor(labels)
    return images, labels


def _worker_fn(worker_id, dataset_factory, indices, staging_queue, stop_event):
    dataset = dataset_factory()
    for idx in indices:
        if stop_event.is_set():
            break
        sample = dataset[idx]
        staging_queue.put(sample)


def _orchestrator_fn(staging_queue, batch_queue, batch_size, stop_event, result_queue):
    buffer = []
    batches_produced = 0
    empty_events = 0
    wait_time = 0.0

    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            sample = staging_queue.get(timeout=0.1)
            wait_time += time.perf_counter() - t0
            buffer.append(sample)
            if len(buffer) >= batch_size:
                batch_samples = buffer[:batch_size]
                buffer = buffer[batch_size:]
                batch = _default_collate_fn(batch_samples)
                batch_queue.put(batch)
                batches_produced += 1
        except:
            empty_events += 1
            continue

    while len(buffer) >= batch_size:
        batch = _default_collate_fn(buffer[:batch_size])
        buffer = buffer[batch_size:]
        batch_queue.put(batch)
        batches_produced += 1

    result_queue.put({
        "batches_produced": batches_produced,
        "empty_events": empty_events,
        "wait_time": wait_time,
    })


class DataLoaderMP:
    def __init__(self, dataset_factory, total_samples, batch_size,
                 num_workers=1, max_staging_size=256, max_batch_queue_size=8,
                 metrics_dir=None):
        self.dataset_factory = dataset_factory
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metrics_dir = metrics_dir

        self._ctx = mp.get_context('spawn')
        self.staging_queue = self._ctx.Queue(maxsize=max_staging_size)
        self.batch_queue = self._ctx.Queue(maxsize=max_batch_queue_size)
        self.result_queue = self._ctx.Queue()

        self.sampler = LockFreeSampler(total_samples, num_workers, shuffle=True)
        self._stop_event = self._ctx.Event()
        self._workers = []
        self._orchestrator = None

    def __iter__(self):
        self._stop_event.clear()
        self._workers = []

        for w in range(self.num_workers):
            indices = self.sampler.get_partition(w)
            p = self._ctx.Process(
                target=_worker_fn,
                args=(w, self.dataset_factory, indices,
                      self.staging_queue, self._stop_event)
            )
            p.start()
            self._workers.append(p)

        self._orchestrator = self._ctx.Process(
            target=_orchestrator_fn,
            args=(self.staging_queue, self.batch_queue, self.batch_size,
                  self._stop_event, self.result_queue)
        )
        self._orchestrator.start()
        return self

    def __next__(self):
        procs = self._workers + ([self._orchestrator] if self._orchestrator else [])
        all_done = all(not p.is_alive() for p in procs)
        if self.batch_queue.empty() and all_done:
            self._cleanup()
            raise StopIteration
        try:
            return self.batch_queue.get(timeout=1.0)
        except:
            if self.batch_queue.empty() and all(not p.is_alive() for p in procs):
                self._cleanup()
                raise StopIteration
            return self.batch_queue.get()

    def _cleanup(self):
        self._stop_event.set()
        for p in self._workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        if self._orchestrator and self._orchestrator.is_alive():
            self._orchestrator.terminate()
            self._orchestrator.join(timeout=2.0)
        try:
            stats = self.result_queue.get_nowait()
        except:
            stats = {"batches_produced": 0, "empty_events": 0, "wait_time": 0}
        if self.metrics_dir:
            os.makedirs(self.metrics_dir, exist_ok=True)
            path = os.path.join(self.metrics_dir,
                                f"metrics_mp_w{self.num_workers}_bs{self.batch_size}.json")
            with open(path, 'w') as f:
                json.dump(stats, f, indent=2)

    def set_epoch(self, seed=None):
        self.sampler.reshuffle(seed)
