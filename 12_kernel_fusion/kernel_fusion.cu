// CUDA Kernel Fusion Example
// This demonstrates fusing multiple operations into a single kernel
// to reduce memory bandwidth and improve performance

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Non-fused approach: Three separate kernels
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void multiply_kernel(float *c, float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = c[idx] * d[idx];
    }
}

__global__ void sqrt_kernel(float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(c[idx]);
    }
}

// Fused kernel: Combines all three operations
__global__ void fused_add_mul_sqrt_kernel(float *a, float *b, float *d, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform all operations in registers without intermediate memory accesses
        float temp = a[idx] + b[idx];  // Addition
        temp = temp * d[idx];           // Multiplication
        result[idx] = sqrtf(temp);      // Square root
    }
}

// More complex fusion example: GEMM + bias + activation
__global__ void fused_gemm_bias_relu(
    float *A, float *B, float *C, float *bias,
    int M, int N, int K, float alpha, float beta)
{
    // Shared memory for tile-based computation
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Tile-based matrix multiplication
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tiles into shared memory
        if (row < M && t * 16 + tx < K)
            As[ty][tx] = A[row * K + t * 16 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t * 16 + ty < K && col < N)
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Apply scaling, bias, and ReLU activation (fused)
    if (row < M && col < N) {
        sum = alpha * sum + beta * C[row * N + col];  // Scaling
        sum += bias[col];                              // Bias addition
        sum = fmaxf(sum, 0.0f);                       // ReLU activation
        C[row * N + col] = sum;
    }
}

// Reduction + normalization fusion example
__global__ void fused_mean_variance_normalize(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Compute sum for mean
    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / n;
    __syncthreads();
    
    // Phase 2: Compute variance
    float var = 0.0f;
    if (idx < n) {
        float diff = input[idx] - mean;
        var = diff * diff;
    }
    sdata[tid] = var;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float variance = sdata[0] / n;
    float stddev = sqrtf(variance + 1e-8f);  // Add epsilon for numerical stability
    
    // Phase 3: Normalize (fused with computation)
    if (idx < n) {
        output[idx] = (input[idx] - mean) / stddev;
    }
}

// Host function to demonstrate usage
void demonstrate_kernel_fusion() {
    const int N = 1024 * 1024;  // 1M elements
    const int bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_d = (float*)malloc(bytes);
    float *h_result = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_d[i] = 3.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_d, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_d, bytes);
    cudaMalloc(&d_result, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Non-fused approach
    cudaEventRecord(start);
    add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    multiply_kernel<<<gridSize, blockSize>>>(d_c, d_d, N);
    sqrt_kernel<<<gridSize, blockSize>>>(d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float nonFusedTime;
    cudaEventElapsedTime(&nonFusedTime, start, stop);
    
    // Fused approach
    cudaEventRecord(start);
    fused_add_mul_sqrt_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_d, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fusedTime;
    cudaEventElapsedTime(&fusedTime, start, stop);
    
    printf("Non-fused time: %.3f ms\n", nonFusedTime);
    printf("Fused time: %.3f ms\n", fusedTime);
    printf("Speedup: %.2fx\n", nonFusedTime / fusedTime);
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_d);
    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    demonstrate_kernel_fusion();
    return 0;
}
