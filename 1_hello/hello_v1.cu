#include <stdio.h>

// Kernel function to generate thread data (no header printing here)
__global__ void generateThreadTable() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    printf("| %-10d | %-11d | %-10d | %-8d |\n",
           blockIdx.x, threadIdx.x, blockDim.x, threadId);
}

int main() {
    // Print table header from CPU (host)
    printf("| blockIdx.x | threadIdx.x | blockDim.x | threadId |\n");
    printf("|------------|-------------|------------|----------|\n");

    // Launch kernel
    generateThreadTable<<<2, 4>>>();

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    return 0;
}

