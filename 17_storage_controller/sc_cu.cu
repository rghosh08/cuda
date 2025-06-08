#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <chrono>

constexpr size_t DATA_SIZE = 1000000000;
constexpr int BLOCK_SIZE = 256; // Number of threads per CUDA block

__global__ void storage_controller_kernel(int *data, int *output, int *checksums, size_t size) {
    __shared__ int local_checksum[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_checksum = 0;

    if (idx < size) {
        output[idx] = data[idx] + 1;
        thread_checksum = output[idx];
    }

    local_checksum[threadIdx.x] = thread_checksum;
    __syncthreads();

    // Reduce checksum within block
    if (threadIdx.x == 0) {
        int checksum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            checksum ^= local_checksum[i];
        }
        checksums[blockIdx.x] = checksum;
    }
}

int main() {
    int *h_data, *h_output;
    int *d_data, *d_output, *d_checksums;

    size_t bytes = DATA_SIZE * sizeof(int);
    int grid_size = (DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host memory allocation
    h_data = new int[DATA_SIZE];
    h_output = new int[DATA_SIZE];
    int *h_checksums = new int[grid_size];

    std::iota(h_data, h_data + DATA_SIZE, 0);

    // Device memory allocation
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_checksums, grid_size * sizeof(int));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    auto start_time = std::chrono::high_resolution_clock::now();

    storage_controller_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, d_output, d_checksums, DATA_SIZE);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_checksums, d_checksums, grid_size * sizeof(int), cudaMemcpyDeviceToHost);


    int final_checksum = 0;
    for (int i = 0; i < grid_size; ++i) {
        final_checksum ^= h_checksums[i];
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << "Data[" << i << "]: " << h_data[i] << " -> " << h_output[i] << '\n';
    }

    std::cout << "Final checksum: " << final_checksum << '\n';

    // Free memory
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_checksums);
    delete[] h_data;
    delete[] h_output;
    delete[] h_checksums;

    return 0;
}

