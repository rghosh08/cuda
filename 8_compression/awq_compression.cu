#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SIZE 4096

// Correct atomic max for floating point numbers
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

// Kernel to compute activation-aware scale factor
__global__ void compute_scale(const float* activations, float* scale) {
    __shared__ float max_val;
    if (threadIdx.x == 0) max_val = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float local_max = 0.0f;
    for (int i = idx; i < SIZE; i += stride)
        local_max = fmaxf(local_max, fabsf(activations[i]));

    atomicMaxFloat(&max_val, local_max);

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0)
        scale[0] = max_val / 127.0f;
}

int main() {
    size_t size = SIZE * sizeof(float);

    float *d_activations, *d_scale;
    cudaMalloc(&d_activations, size);
    cudaMalloc(&d_scale, sizeof(float));

    // Initialize activations (dummy data)
    float activations[SIZE];
    for (int i = 0; i < SIZE; ++i)
        activations[i] = sinf(i);

    cudaMemcpy(d_activations, activations, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    compute_scale<<<16, 256>>>(d_activations, d_scale);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float h_scale;
    cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Scale factor: %.6f\n", h_scale);
    printf("Time taken: %.4f milliseconds\n", milliseconds);

    // Cleanup
    cudaFree(d_activations);
    cudaFree(d_scale);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

