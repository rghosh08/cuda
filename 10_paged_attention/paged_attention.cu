#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SEQ_LEN 128
#define PAGE_SIZE 32
#define EMBED_DIM 64
#define NUM_PAGES ((SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE)

// Kernel for paged attention computation
__global__ void paged_attention(float *Q, float *K, float *V, float *output, int seq_len, int embed_dim) {
    extern __shared__ float shared_mem[];
    float *shared_K = shared_mem;
    float *shared_V = &shared_mem[PAGE_SIZE * embed_dim];

    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len) return;

    float attention_out[EMBED_DIM] = {0.0f};

    for (int page = 0; page < NUM_PAGES; ++page) {
        int page_start = page * PAGE_SIZE;
        int k_idx = page_start + threadIdx.x;

        // Load keys and values into shared memory
        if (k_idx < seq_len) {
            for (int d = 0; d < embed_dim; d++) {
                shared_K[threadIdx.x * embed_dim + d] = K[k_idx * embed_dim + d];
                shared_V[threadIdx.x * embed_dim + d] = V[k_idx * embed_dim + d];
            }
        }
        __syncthreads();

        // Compute attention within this page
        for (int k = 0; k < PAGE_SIZE && (page_start + k) < seq_len; k++) {
            float score = 0.0f;
            for (int d = 0; d < embed_dim; d++)
                score += Q[q_idx * embed_dim + d] * shared_K[k * embed_dim + d];

            score = expf(score / sqrtf(embed_dim));

            for (int d = 0; d < embed_dim; d++)
                attention_out[d] += score * shared_V[k * embed_dim + d];
        }
        __syncthreads();
    }

    // Write final attention output
    for (int d = 0; d < embed_dim; d++)
        output[q_idx * embed_dim + d] = attention_out[d];
}

int main() {
    size_t tensor_size = SEQ_LEN * EMBED_DIM * sizeof(float);

    float *h_Q = new float[SEQ_LEN * EMBED_DIM];
    float *h_K = new float[SEQ_LEN * EMBED_DIM];
    float *h_V = new float[SEQ_LEN * EMBED_DIM];
    float *h_output = new float[SEQ_LEN * EMBED_DIM];

    // Initialize input tensors
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        h_Q[i] = 0.01f; h_K[i] = 0.02f; h_V[i] = 0.03f;
    }

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, tensor_size);
    cudaMalloc(&d_K, tensor_size);
    cudaMalloc(&d_V, tensor_size);
    cudaMalloc(&d_output, tensor_size);

    cudaMemcpy(d_Q, h_Q, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, tensor_size, cudaMemcpyHostToDevice);

    dim3 threads(PAGE_SIZE);
    dim3 blocks((SEQ_LEN + PAGE_SIZE - 1) / PAGE_SIZE);

    size_t shared_mem_size = 2 * PAGE_SIZE * EMBED_DIM * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    paged_attention<<<blocks, threads, shared_mem_size>>>(d_Q, d_K, d_V, d_output, SEQ_LEN, EMBED_DIM);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, tensor_size, cudaMemcpyDeviceToHost);

    printf("Output sample [0]: %.4f\n", h_output[0]);
    printf("Output sample [1]: %.4f\n", h_output[1]);
    printf("Execution Time: %.4f milliseconds\n", milliseconds);

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

