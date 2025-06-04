#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute distances between query and nodes
__global__ void compute_distances(float *nodes, float *query, float *distances, int dim, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        float distance = 0;
        for (int d = 0; d < dim; d++) {
            float diff = nodes[idx * dim + d] - query[d];
            distance += diff * diff;
        }
        distances[idx] = sqrtf(distance);
    }
}

int main() {
    const int num_nodes = 1000000;
    const int dim = 128;

    float *nodes = (float*)malloc(num_nodes * dim * sizeof(float));
    float *query = (float*)malloc(dim * sizeof(float));
    float *distances = (float*)malloc(num_nodes * sizeof(float));

    // Random initialization
    for (int i = 0; i < num_nodes * dim; i++) nodes[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < dim; i++) query[i] = rand() / (float)RAND_MAX;

    float *d_nodes, *d_query, *d_distances;
    cudaMalloc(&d_nodes, num_nodes * dim * sizeof(float));
    cudaMalloc(&d_query, dim * sizeof(float));
    cudaMalloc(&d_distances, num_nodes * sizeof(float));

    cudaMemcpy(d_nodes, nodes, num_nodes * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    compute_distances<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_query, d_distances, dim, num_nodes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(distances, d_distances, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print sample distances
    printf("Sample distances computed:\n");
    for (int i = 0; i < 5; i++) {
        printf("Node %d: %.4f\n", i, distances[i]);
    }

    printf("Computation Time: %.3f ms\n", milliseconds);

    cudaFree(d_nodes);
    cudaFree(d_query);
    cudaFree(d_distances);
    free(nodes);
    free(query);
    free(distances);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
