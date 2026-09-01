#!/usr/bin/env python3
"""Test PyTorch DataLoader with our ParquetDataset."""
import sys, time
sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset
from torch.utils.data import DataLoader as TorchDataLoader
import torch

def collate_fn(samples):
    images = torch.stack([s[0] for s in samples])
    labels = [s[1] for s in samples]
    if isinstance(labels[0], str):
        labels = torch.tensor([int(l.replace('n', '')) for l in labels])
    else:
        labels = torch.tensor(labels)
    return images, labels

dataset = ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=10000)
print(f"Dataset: {len(dataset)} samples", flush=True)

for nw in [1, 2, 4, 8, 16, 32]:
    loader = TorchDataLoader(dataset, batch_size=256, num_workers=nw, collate_fn=collate_fn)
    batch_count = 0
    start = time.time()
    for batch in loader:
        batch_count += 1
    elapsed = time.time() - start
    print(f"PYTORCH_PARQUET,{nw},{batch_count},{elapsed:.2f},{batch_count*256/elapsed:.1f}", flush=True)

print("DONE")
