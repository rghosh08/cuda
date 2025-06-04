#include <cuda_runtime.h>
#include <iostream>
#include <climits>

#define N 1024

__global__ void findMinEdge(int *graph, bool *inMST, int *minEdge, int *minIndex) {
    int tid = threadIdx.x;
    if (tid < N && !inMST[tid]) {
        int min = INT_MAX;
        for (int i = 0; i < N; i++) {
            if (inMST[i] && graph[i * N + tid] < min) {
                min = graph[i * N + tid];
            }
        }
        minEdge[tid] = min;
        minIndex[tid] = tid;
    } else {
        minEdge[tid] = INT_MAX;
        minIndex[tid] = -1;
    }
}

int main() {
    int graph[N * N];
    bool inMST[N] = {false};

    // Random initialization for the adjacency matrix
    for (int i = 0; i < N * N; i++) {
        graph[i] = rand() % 100 + 1;
    }

    inMST[0] = true;

    int *d_graph, *d_minEdge, *d_minIndex;
    bool *d_inMST;

    cudaMalloc(&d_graph, N * N * sizeof(int));
    cudaMalloc(&d_inMST, N * sizeof(bool));
    cudaMalloc(&d_minEdge, N * sizeof(int));
    cudaMalloc(&d_minIndex, N * sizeof(int));

    cudaMemcpy(d_graph, graph, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inMST, inMST, N * sizeof(bool), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int mst_weight = 0;

    for (int count = 1; count < N; count++) {
        findMinEdge<<<1, N>>>(d_graph, d_inMST, d_minEdge, d_minIndex);

        int minEdge[N], minIndex[N];
        cudaMemcpy(minEdge, d_minEdge, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(minIndex, d_minIndex, N * sizeof(int), cudaMemcpyDeviceToHost);

        int min_weight = INT_MAX;
        int min_node = -1;

        for (int i = 0; i < N; i++) {
            if (minEdge[i] < min_weight) {
                min_weight = minEdge[i];
                min_node = minIndex[i];
            }
        }

        mst_weight += min_weight;
        inMST[min_node] = true;

        cudaMemcpy(d_inMST, inMST, N * sizeof(bool), cudaMemcpyHostToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Minimum Spanning Tree Weight: " << mst_weight << "\n";
    std::cout << "Computation Time: " << milliseconds << " ms\n";


    cudaFree(d_graph);
    cudaFree(d_inMST);
    cudaFree(d_minEdge);
    cudaFree(d_minIndex);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

