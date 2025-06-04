#include <stdio.h>

// Kernel function to generate a markdown-formatted table showing thread indices
__global__ void generateThreadTable() {
    // Calculate global thread ID
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    // Print table header only from the first thread to avoid duplicate headers
    if (threadId == 0) {
        printf("| blockIdx.x | threadIdx.x | blockDim.x | threadId |\n");
        printf("|------------|-------------|------------|----------|\n");
    }

    // Synchronize all threads to ensure the header prints before any thread data
    __syncthreads();

    // Each thread prints its own indices and global ID in table format
    printf("| %-10d | %-11d | %-10d | %-8d |\n",
           blockIdx.x, threadIdx.x, blockDim.x, threadId);
}

int main() {
    // Launch kernel with 2 blocks, each containing 4 threads
    generateThreadTable<<<2, 4>>>();

    // Wait for all GPU tasks to complete before ending the program
    cudaDeviceSynchronize();

    return 0;
}
