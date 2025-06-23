#!/bin/bash

# Clean up any previous builds
rm -f paged_flash_attention_cuda

# Compile both source files together
nvcc -std=c++14 -O3 -arch=sm_86 \
    main_cuda.cu \
    paged_flash_attention_cuda.cu \
    -o paged_flash_attention_cuda

echo "Compilation complete!"
