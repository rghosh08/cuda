#include <cuda_runtime.h>
#include <cstdio>

#define HEAD_DIM 64
#define NUM_HEADS 16
#define MAX_SEQ_LEN 2048


// Cache shape: [num_heads, max_seq_len, head_dim]

// Insert keys and values at a specific position (seq_pos)

__global__ void insert_kv_cache(float *cache_k, float *cache_v, const float *new_k, const float *new_v, int seq_pos) {
    int head = blockIdx.x;
    int dim = threadIdx.x;

    if (head < NUM_HEADS && dim < HEAD_DIM) {
        int idx = head * MAX_SEQ_LEN * HEAD_DIM + seq_pos * HEAD_DIM + dim;
        cache_k[idx] = new_k[head * HEAD_DIM + dim];
        cache_v[idx] = new_v[head * HEAD_DIM + dim];
    }
}

// Retrieve keys and values up to the current sequence position
__global__ void retrieve_kv_cache(float *cache_k, float *cache_v, float *output_k, float *output_v, int current_seq_len) {
    int head = blockIdx.x;
    int seq_pos = threadIdx.y;
    int dim = threadIdx.x;

    if (head < NUM_HEADS && dim < HEAD_DIM && seq_pos < current_seq_len) {
        int cache_idx = head * MAX_SEQ_LEN * HEAD_DIM + seq_pos * HEAD_DIM + dim;
        int out_idx = head * current_seq_len * HEAD_DIM + seq_pos * HEAD_DIM + dim;

        output_k[out_idx] = cache_k[cache_idx];
        output_v[out_idx] = cache_v[cache_idx];
    }
}

int main() {
    const int seq_len = 5;  // realistic decoding step example
    size_t cache_size = NUM_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t kv_step_size = NUM_HEADS * HEAD_DIM * sizeof(float);
    size_t kv_full_size = NUM_HEADS * seq_len * HEAD_DIM * sizeof(float);

    // Allocate KV-cache GPU memory
    float *cache_k, *cache_v;
    cudaMalloc(&cache_k, cache_size);
    cudaMalloc(&cache_v, cache_size);
    cudaMemset(cache_k, 0, cache_size);
    cudaMemset(cache_v, 0, cache_size);

    // Simulate insertion over multiple decoding steps
    for (int seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
        // Prepare new K/V (simulated output from transformer layer)
        float host_new_k[NUM_HEADS * HEAD_DIM];
        float host_new_v[NUM_HEADS * HEAD_DIM];
        for (int i = 0; i < NUM_HEADS * HEAD_DIM; ++i) {
            host_new_k[i] = seq_pos + 0.1f;
            host_new_v[i] = seq_pos + 0.2f;
        }

        float *device_new_k, *device_new_v;
        cudaMalloc(&device_new_k, kv_step_size);
        cudaMalloc(&device_new_v, kv_step_size);

        cudaMemcpy(device_new_k, host_new_k, kv_step_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_new_v, host_new_v, kv_step_size, cudaMemcpyHostToDevice);

        // Insert new KV into cache
        insert_kv_cache<<<NUM_HEADS, HEAD_DIM>>>(cache_k, cache_v, device_new_k, device_new_v, seq_pos);
        cudaDeviceSynchronize();

        cudaFree(device_new_k);
        cudaFree(device_new_v);
    }

    // Retrieve the full KV cache after multiple steps
    float *output_k, *output_v;
    cudaMalloc(&output_k, kv_full_size);
    cudaMalloc(&output_v, kv_full_size);

    dim3 threads(HEAD_DIM, seq_len); // (dim, seq_len)
    retrieve_kv_cache<<<NUM_HEADS, threads>>>(cache_k, cache_v, output_k, output_v, seq_len);
    cudaDeviceSynchronize();

    // Copy results back for verification
    float host_output_k[NUM_HEADS * seq_len * HEAD_DIM];
    float host_output_v[NUM_HEADS * seq_len * HEAD_DIM];

    cudaMemcpy(host_output_k, output_k, kv_full_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_output_v, output_v, kv_full_size, cudaMemcpyDeviceToHost);

    // Print to verify correctness
    for (int pos = 0; pos < seq_len; ++pos) {
        printf("Position %d, Head 0, K[0]: %.2f, V[0]: %.2f\n",
               pos,
               host_output_k[pos * HEAD_DIM],
               host_output_v[pos * HEAD_DIM]);
    }

    // Cleanup
    cudaFree(cache_k);
    cudaFree(cache_v);
    cudaFree(output_k);
    cudaFree(output_v);

    return 0;
}
