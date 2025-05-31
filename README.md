*CUDA Programming 


**Thread Hierarchy

Thread: Smallest execution unit
Warp: 32 threads executing in SIMD fashion
Block: Group of threads (up to 1024) that can cooperate
Grid: Collection of blocks

Grid (2D or 3D)
├── Block (0,0)
│   ├── Thread (0,0,0)
│   ├── Thread (1,0,0)
│   └── ...
├── Block (1,0)
└── ...
