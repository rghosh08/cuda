#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

__global__ void compute_distances(float *nodes, float *query, float *distances, int dim, size_t num_nodes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        float distance = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = nodes[idx * dim + d] - query[d];
            distance += diff * diff;
        }
        distances[idx] = sqrtf(distance);
    }
}

void printGPUMemoryUsage(const char* stage) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double used_gb = (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    printf("%s GPU Memory: %.2f GB used / %.2f GB total\n", stage, used_gb, total_gb);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_nodes> <dim>\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t num_nodes = atoll(argv[1]);
    int dim = atoi(argv[2]);

    printf("Allocating host memory...\n");
    float *nodes = (float*)malloc(num_nodes * dim * sizeof(float));
    float *query = (float*)malloc(dim * sizeof(float));
    float *distances = (float*)malloc(num_nodes * sizeof(float));

    printf("Initializing data...\n");
    for (size_t i = 0; i < num_nodes * dim; i++) nodes[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < dim; i++) query[i] = rand() / (float)RAND_MAX;

    float *d_nodes, *d_query, *d_distances;

    printf("Allocating GPU memory...\n");
    cudaMalloc(&d_nodes, num_nodes * dim * sizeof(float));
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_distances, num_nodes * sizeof(float));

    printf("Transferring data to GPU...\n");
    cudaMemcpy(d_nodes, nodes, num_nodes * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice);

    printGPUMemoryUsage("After allocations & transfers");

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

    printf("Starting CUDA kernel execution...\n");
    auto start = std::chrono::high_resolution_clock::now();

    compute_distances<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_query, d_distances, dim, num_nodes);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("CUDA kernel execution completed.\n");
    printf("Kernel Execution Time: %.3f ms\n", elapsed.count());

    printGPUMemoryUsage("After computation");

    printf("Transferring results back to host...\n");
    cudaMemcpy(distances, d_distances, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sample distances computed:\n");
    for (int i = 0; i < 5; i++)
        printf("Node %d: %.4f\n", i, distances[i]);

    printf("Cleaning up memory...\n");
    free(nodes);
    free(query);
    free(distances);
    cudaFree(d_nodes);
    cudaFree(d_query);
    cudaFree(d_distances);

    return 0;
}

