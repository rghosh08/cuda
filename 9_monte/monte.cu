#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>

#define THREADS 256
#define BLOCKS 256
#define TOTAL_POINTS (THREADS * BLOCKS * 1000)

// GPU kernel: Monte Carlo simulation
__global__ void monte_carlo_pi(int *count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(1234, id, 0, &state);

    int local_count = 0;
    for (int i = 0; i < 1000; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x*x + y*y <= 1.0f) local_count++;
    }
    count[id] = local_count;
}

int main() {
    int *d_count, h_count[THREADS * BLOCKS];

    cudaMalloc(&d_count, THREADS * BLOCKS * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    monte_carlo_pi<<<BLOCKS, THREADS>>>(d_count);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_count, d_count, THREADS * BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    int total_in_circle = 0;
    for (int i = 0; i < THREADS * BLOCKS; i++)
        total_in_circle += h_count[i];

    float pi_estimate = (4.0f * total_in_circle) / TOTAL_POINTS;
    printf("Estimated Pi = %f\n", pi_estimate);
    printf("Time taken: %.4f milliseconds\n", milliseconds);

    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

