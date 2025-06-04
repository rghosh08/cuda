# CUDA Programming 
"People who are really serious about software should make their own hardware." Alan Kay

## Thread Hierarchy

* Thread: Smallest execution unit
* Warp: 32 threads executing in SIMD fashion
* Block: Group of threads (up to 1024) that can cooperate
* Grid: Collection of blocks
  
```
Grid (2D or 3D)
├── Block (0,0)
│   ├── Thread (0,0,0)
│   ├── Thread (1,0,0)
│   └── ...
├── Block (1,0)
└── ...
```

## Main Reasons for CUDA's Speed:

* Massive Parallelism: GPUs have thousands of simple cores vs CPUs with dozens of complex cores. This allows processing thousands of data elements simultaneously.
* SIMD Architecture: Each instruction operates on 32 threads at once (a warp), providing 32x throughput per instruction.
* High Memory Bandwidth: GPUs have 5-10x higher memory bandwidth than CPUs (up to 1.2 TB/s vs 100 GB/s).
* Hardware Thread Scheduling: Zero-overhead context switching between warps keeps the GPU cores busy while waiting for memory.
* Specialized Hardware: Dedicated units for math operations, atomics, and tensor operations.
Memory Hierarchy: Multiple levels of increasingly faster memory (registers → shared memory → L1/L2 → global).
* Coalesced Memory Access: Combining multiple thread memory requests into single transactions.

## The Speed in Numbers:

* A modern GPU can have 10,000+ cores vs 64 CPU cores
* Memory bandwidth: 900+ GB/s vs 100 GB/s
* Matrix multiplication: 100-1000x faster
* Image processing: 50-100x faster
* Deep learning training: 10-50x faster

## GPU-CPU Interaction

 ![cpu_gpu_interaction](/assets/images/cpu_gpu_interactions.jpg)
