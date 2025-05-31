#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SIZE 4096  // number of weights per layer

// Compute activation-aware scale factor
__global__ void compute_scale(float* activations, float* scale) {
    __shared__ float max_val;
    max_val = 0.0f;

    int idx = threadIdx.x;
    float val = fabsf(activations[idx]);
    atomicMax((int*)&max_val, __float_as_int(val));

    __syncthreads();

    if (idx == 0) {
        scale[0] = max_val / 127.0f;  // INT8 max range [-127,127]
    }
}

// Quantize weights using computed scale
__global__ void quantize_weights(float* weights_fp32, int8_t* weights_int8, float* scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        float scaled_weight = weights_fp32[idx] / scale[0];
        int quantized = roundf(scaled_weight);
        quantized = max(min(quantized, 127), -127);
        weights_int8[idx] = (int8_t)quantized;
    }
}

// Dequantize (for verification)
__global__ void dequantize_weights(int8_t* weights_int8, float* weights_dequantized, float* scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SIZE) {
        weights_dequantized[idx] = weights_int8[idx] * scale[0];
    }
}

int main() {
    size_t weight_size_fp32 = SIZE * sizeof(float);
    size_t weight_size_int8 = SIZE * sizeof(int8_t);

    // Allocate host weights and activations
    float h_weights[SIZE], h_activations[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_weights[i] = sinf(i);          // simulate weights
        h_activations[i] = cosf(i);      // simulate activations
    }

    // Allocate device memory
    float *d_weights_fp32, *d_activations, *d_scale, *d_weights_dequantized;
    int8_t *d_weights_int8;

    cudaMalloc(&d_weights_fp32, weight_size_fp32);
    cudaMalloc(&d_weights_int8, weight_size_int8);
    cudaMalloc(&d_weights_dequantized, weight_size_fp32);
    cudaMalloc(&d_activations, weight_size_fp32);
    cudaMalloc(&d_scale, sizeof(float));

    cudaMemcpy(d_weights_fp32, h_weights, weight_size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_activations, h_activations, weight_size_fp32, cudaMemcpyHostToDevice);

    // Compute activation-aware scale
    compute_scale<<<1, SIZE>>>(d_activations, d_scale);
    cudaDeviceSynchronize();

    // Quantize weights
    quantize_weights<<<(SIZE+255)/256, 256>>>(d_weights_fp32, d_weights_int8, d_scale);
    cudaDeviceSynchronize();

    // Dequantize weights for verification
    dequantize_weights<<<(SIZE+255)/256, 256>>>(d_weights_int8, d_weights_dequantized, d_scale);
    cudaDeviceSynchronize();

    // Verify results
    float h_scale;
    cudaMemcpy(&h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost);

    float h_dequantized[SIZE];
    cudaMemcpy(h_dequantized, d_weights_dequantized, weight_size_fp32, cudaMemcpyDeviceToHost);

    printf("Scale factor: %.6f\n", h_scale);
    printf("Original Weight[0]: %.6f, Quant-Dequant Weight[0]: %.6f\n", h_weights[0], h_dequantized[0]);
    printf("Original Weight[100]: %.6f, Quant-Dequant Weight[100]: %.6f\n", h_weights[100], h_dequantized[100]);

    // Cleanup
    cudaFree(d_weights_fp32);
    cudaFree(d_weights_int8);
    cudaFree(d_weights_dequantized);
    cudaFree(d_activations);
    cudaFree(d_scale);

    return 0;
}

