#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 512

// CUDA Kernel: Compute simple checksum
__global__ void computeChecksum(unsigned char* data, size_t size, unsigned int* checksum) {
    __shared__ unsigned int shared_sum[BLOCK_SIZE];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    shared_sum[tid] = (idx < size) ? data[idx] : 0;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride)
            shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    // Atomic add block result to global checksum
    if (tid == 0)
        atomicAdd(checksum, shared_sum[0]);
}

int main() {
    const size_t dataSize = 1 << 24; // 16 MB
    unsigned char* h_data = (unsigned char*)malloc(dataSize);

    // Initialize data (simulate read from storage)
    for (size_t i = 0; i < dataSize; ++i)
        h_data[i] = rand() % 256;

    unsigned char* d_data;
    unsigned int* d_checksum;
    unsigned int h_checksum = 0;

    cudaMalloc(&d_data, dataSize);
    cudaMalloc(&d_checksum, sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_checksum, 0, sizeof(unsigned int));

    dim3 block(BLOCK_SIZE);
    dim3 grid((dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    computeChecksum<<<grid, block>>>(d_data, dataSize, d_checksum);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_checksum, d_checksum, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Checksum: %u\n", h_checksum);
    printf("Processing Time: %.4f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_checksum);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

