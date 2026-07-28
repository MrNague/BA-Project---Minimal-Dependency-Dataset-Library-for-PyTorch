# Minimal Dependency Dataset Library for PyTorch

Bachelor Project — DFKI (German Research Center for Artificial Intelligence) | SS 2026

## Project Description

Training large deep learning models on GPU clusters requires efficient data pipelines. When datasets reside on remote network storage, loading and preprocessing can become the primary bottleneck, leaving expensive GPUs underutilized. Existing solutions (WebDataset, MosaicML StreamingDataset, TFRecord) often require extensive dependencies and major changes to training code.

The goal of this project is to analyse performance bottlenecks in the data loading pipeline, then design and implement a simple library for storing and loading training data from a remote storage server. The library should be minimal — relying only on PyTorch and the Python standard library — while supporting multi-threaded loading, lock-free sampling, distributed training (DDP), and built-in instrumentation for performance profiling.

## Project Goals & Status

| # | Goal | Status |
|---|------|--------|
| 1 | Train baseline model + profile data loading bottlenecks | Done |
| 2 | Benchmark GPU compute delays across 10 GPU types (3,540 runs) | Done |
| 3 | Build GPU simulator (Dummy Model) for GPU-free testing | Done |
| 4 | Evaluate 6 storage formats for throughput, overhead, thread safety | Done |
| 5 | Implement multi-threaded DataLoader with lock-free sampler | Done |
| 6 | Add built-in instrumentation (MonitoredQueue + MetricsTracker) | Done |
| 7 | Run benchmark sweep: 9 batch sizes × 6 worker counts, 100k images | Done |
| 8 | Compare against PyTorch DataLoader on same hardware/data | Done |

## Key Results

### Storage Format Evaluation (100k images, tiny-imagenet, fscratch SSD)

| Format | Sequential (img/s) | Random (img/s) | Overhead | Thread-Safe |
|--------|-------------------|----------------|----------|-------------|
| Parquet | 472.7 | 473.7 | -0.3% | Yes |
| MessagePack | 431.4 | 312.9 | 0.0% | Conditional |
| Tar | 424.1 | N/A | 1.5% | Sequential only |
| LMDB | 415.8 | 304.7 | 1.8% | Yes |
| Zip | 341.5 | 233.2 | 0.1% | No |
| Plain JPEG | 266.6 | 255.9 | 0% | Yes |

### DataLoader Performance (Parquet, A100 CPU node, 100k images)

| Metric | Our DataLoader | PyTorch DataLoader |
|--------|---------------|-------------------|
| Peak Throughput | 1,787 samples/s (BS=512, 16w) | 2,865 samples/s (BS=512, 32w) |
| Optimal Workers | 16 | 32 |
| Scaling 1-16 workers | 6.5x | 9.8x |
| Scaling 16-32 workers | Degrades (-10%) | Continues (+8%) |
| Memory | 14-17 GB | 16 GB |
| Dependencies | stdlib + torch | PyTorch only (stdli + torch)|

### GPU Compute Delays (ResNet-50, batch_size=256, SGD)

| GPU | Avg Delay (ms) | Models Tested |
|-----|---------------|---------------|
| H200 | 522 | 15 |
| H100 | 597 | 15 |
| RTXB6000 | 537 | 15 |
| L40S | 663 | 11 |
| A100-40GB | 702 | 9 |
| B200 | 711 | 16 |
| RTXA6000 | 1,042 | 9 |

## Repository Structure

bachelor-project/                                                                                                                                                                                                                                                             
├── minimal_dataset/ # Library module                                                                                                                                                                                                                                        
│ ├── init.py # Public API                                                                                                                                                                                                                                                
│ ├── dataset.py # BaseDataset (ImageFolder reader)                                                                                                                                                                                              
│ ├── parquet_dataset.py # ParquetDataset (Parquet reader)                                                                                                                                         
│ ├── sampler.py # LockFreeSampler (lock-free index partitioning)                                                                                                                                                  
│ ├── dataloader.py # DataLoader (multi-threaded, staging + batch queues)                                                                                                                                                  
│ ├── monitored_queue.py # MonitoredQueue (queue instrumentation)                                                                                                                  
│ └── metrics.py # MetricsTracker + WorkerMetrics                                                                                                                                                                                                  
├── benchmarks/                                                                                                                                                                                                                
│ ├── gpu/ # GPU benchmarking & dummy model                                                                                                                                                              
│ │ ├── benchmark_gpu.py                                                                                                                                                                                                                              
│ │ ├── calibrate_dummy.py                                                                                                                                                                                                         
│ │ ├── dummy_model.py                                                                                                                                                                                                                      
│ │ ├── build_delay_config.py                                                                                                                                                                                                                  
│ │ └── gpu_delays.json                                                                                                                                                                               
│ └── storage/ # Storage format evaluation                                                                                                                                                                                                                           
│ ├── benchmark_plain.py                                                                                                                                                                                       
│ ├── benchmark_zip.py                                                                                                                                                                                                                                 
│ ├── benchmark_tar.py                                                                                                                                                                                       
│ ├── benchmark_lmdb.py                                                                                                                                                           
│ ├── benchmark_parquet.py                                                                                                                                                                                                         
│ ├── benchmark_msgpack.py                                                                                                                                                                                                     
│ └── file_io.py                                                                                                                                                                                       
├── tests/ # DataLoader tests & benchmarks                                                                                                                                                                                                  
│ ├── test_dataloader.py                                                                                                                                                                                                                            
│ ├── benchmark_dataloader.py                                                                                                                                                                                                                                                 
│ ├── benchmark_pytorch.py                                                                                                                                                                                                                                                   
│ ├── plot_comparison.py                                                                                                                                                                                                                         
│ └── launch_full_benchmark.sh                                                                                                                                                                                                         
├── training/ # ResNet training scripts                                                                                                                                                                                                                            
│ ├── train_resnet50.py                                                                                                                                                                                                                                   
│ ├── train_resnet50_ddp.py                                                                                                                                                                                                     
│ └── analyze_dataloading.py                                                                                                                                                                                                                           
├── docs/                                                                                                                                                                                                                                                  
│ ├── images/ # Architecture diagrams & benchmark plots                                                                                                                                                                                                                  
│ │ ├── 00_MainOverview1.png                                                                                                                                                                                                                      
│ │ ├── 02_dataLoader1.png                                                                                                                                                                                                                                                    
│ │ ├── 04_Storage Format Evaluation.png                                                                                                                                                                                            
│ │ ├── 05_GPU Compute Delay.png                                                                                                                                                                                                                
│ │ ├── plot_ours_throughput.png                                                                                                                                                                                                                           
│ │ ├── plot_pytorch_throughput.png                                                                                                                                                                                                                    
│ │ ├── plot_comparison_bs16.png                                                                                                                                                                                                                        
│ │ └── plot_speedup_comparison.png                                                                                                                                                                                                              
│ └── architecture.md                                                                                                                                                                       
└── README.md                                                                                                                 
