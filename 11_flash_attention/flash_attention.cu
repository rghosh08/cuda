#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_SIZE 16
#define SEQ_LEN 128
#define HEAD_DIM 64

// CUDA kernel implementing FlashAttention-like computation
__global__ void flash_attention(
    const float *Q, const float *K, const float *V, float *output, int seq_len, int head_dim) {

    extern __shared__ float shared_mem[];
    float *shared_Q = shared_mem;
    float *shared_K = shared_mem + TILE_SIZE * head_dim;
    float *shared_V = shared_mem + 2 * TILE_SIZE * head_dim;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float score = 0.0f;

    for (int tile = 0; tile < seq_len / TILE_SIZE; tile++) {

        // Load tiles into shared memory
        if (row < seq_len && tile * TILE_SIZE + threadIdx.x < seq_len) {
            shared_Q[threadIdx.y * head_dim + threadIdx.x] =
                Q[row * head_dim + tile * TILE_SIZE + threadIdx.x];
            shared_K[threadIdx.y * head_dim + threadIdx.x] =
                K[(tile * TILE_SIZE + threadIdx.y) * head_dim + col];
            shared_V[threadIdx.y * head_dim + threadIdx.x] =
                V[(tile * TILE_SIZE + threadIdx.y) * head_dim + col];
        }
        __syncthreads();

        // Compute scaled dot-product attention
        for (int k = 0; k < TILE_SIZE; k++) {
            float qk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                qk += shared_Q[threadIdx.y * head_dim + d] * shared_K[k * head_dim + d];
            }
            score += expf(qk / sqrtf(head_dim)) * shared_V[k * head_dim + threadIdx.x];
        }
        __syncthreads();
    }

    // Store result
    if (row < seq_len && col < head_dim) {
        output[row * head_dim + col] = score;
    }
}

int main() {
    size_t tensor_size = SEQ_LEN * HEAD_DIM * sizeof(float);

    float *h_Q = new float[SEQ_LEN * HEAD_DIM];
    float *h_K = new float[SEQ_LEN * HEAD_DIM];
    float *h_V = new float[SEQ_LEN * HEAD_DIM];
    float *h_output = new float[SEQ_LEN * HEAD_DIM];

    // Initialize input tensors
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
        h_Q[i] = 0.01f;
        h_K[i] = 0.02f;
        h_V[i] = 0.03f;
    }

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, tensor_size);
    cudaMalloc(&d_K, tensor_size);
    cudaMalloc(&d_V, tensor_size);
    cudaMalloc(&d_output, tensor_size);

    cudaMemcpy(d_Q, h_Q, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, tensor_size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((HEAD_DIM + TILE_SIZE - 1) / TILE_SIZE, (SEQ_LEN + TILE_SIZE - 1) / TILE_SIZE);

    size_t shared_mem_size = 3 * TILE_SIZE * HEAD_DIM * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    flash_attention<<<blocks, threads, shared_mem_size>>>(d_Q, d_K, d_V, d_output, SEQ_LEN, HEAD_DIM);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, tensor_size, cudaMemcpyDeviceToHost);

    printf("Output sample [0]: %.4f\n", h_output[0]);
    printf("Execution time: %.4f ms\n", milliseconds);

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

