#!/usr/bin/env python3
"""Deep instrumentation benchmark."""
import sys, os, time, argparse

sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import ParquetDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, required=True)
args = parser.parse_args()

dataset = ParquetDataset("/fscratch/nague/storage_benchmarks/images.parquet", max_samples=10000)
loader = DataLoader(
    dataset, batch_size=256, num_workers=args.num_workers,
    metrics_dir="/netscratch/nague/deep_metrics"
)

batch_count = 0
start = time.time()
for batch in loader:
    batch_count += 1
elapsed = time.time() - start

summary = loader._cleanup()

if args.num_workers == 1:
    print("workers,batches,time_s,throughput,io_ms,decode_ms,staging_put_ms,collate_ms,batch_put_ms,staging_empty,staging_full,batch_wait_s")

st = summary["stage_times"]
sq = summary["staging_queue"]
bq = summary["batch_queue"]

print(f"{args.num_workers},{batch_count},{elapsed:.2f},{batch_count*256/elapsed:.1f},"
      f"{st['io']['mean_ms']},{st['decode']['mean_ms']},"
      f"{st['staging_put']['mean_ms']},{st['collate']['mean_ms']},{st['batch_put']['mean_ms']},"
      f"{sq['empty_events']},{sq['full_events']},{bq['total_get_wait_s']}")
