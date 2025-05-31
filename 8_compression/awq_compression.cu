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

// Kernel to quantize weights
__global__ void quantize_weights(const float* weights_fp32, int8_t* weights_int8, const float* scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        float scaled = weights_fp32[idx] / scale[0];
        weights_int8[idx] = (int8_t)max(min(roundf(scaled), 127.0f), -127.0f);
    }
}

// Kernel to dequantize weights (for verification)
__global__ void dequantize_weights(const int8_t* weights_int8, float* weights_fp32, const float* scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE)
        weights_fp32[idx] = weights_int8[idx] * scale[0];
}

int main() {
    float h_weights[SIZE], h_activations[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_weights[i] = sinf(i);
        h_activations[i] = cosf(i);
    }

    float *d_weights, *d_activations, *d_scale, *d_dequant;
    int8_t *d_weights_int8;

    cudaMalloc(&d_weights, SIZE * sizeof(float));
    cudaMalloc(&d_activations, SIZE * sizeof(float));
    cudaMalloc(&d_scale, sizeof(float));
    cudaMalloc(&d_weights_int8, SIZE * sizeof(int8_t));
    cudaMalloc(&d_dequant, SIZE * sizeof(float));

    cudaMemcpy(d_weights, h_weights, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_activations, h_activations, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    compute_scale<<<16, 256>>>(d_activations, d_scale);
    cudaDeviceSynchronize();

    quantize_weights<<<16, 256>>>(d_weights, d_weights_int8, d_scale);
    cudaDeviceSynchronize();

    dequantize_weights<<<16, 256>>>(d_weights_int8, d_dequant, d_scale);
    cudaDeviceSynchronize();

    float h_scale, h_dequant[SIZE];
    cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dequant, d_dequant, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Scale factor: %f\n", h_scale);
    printf("Original[0]: %f, Dequantized[0]: %f\n", h_weights[0], h_dequant[0]);
    printf("Original[100]: %f, Dequantized[100]: %f\n", h_weights[100], h_dequant[100]);

    cudaFree(d_weights);
    cudaFree(d_activations);
    cudaFree(d_scale);
    cudaFree(d_weights_int8);
    cudaFree(d_dequant);

    return 0;
}

