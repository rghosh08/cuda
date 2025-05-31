#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SEQ_LEN 64
#define EMBED_DIM 256
#define NUM_HEADS 8
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)

// Compute attention scores: (Q*K^T) per head
__global__ void attention_scores(float* Q, float* K, float* scores) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y;
    int k_idx = threadIdx.x;

    if (head < NUM_HEADS && q_idx < SEQ_LEN && k_idx < SEQ_LEN) {
        float score = 0.0f;
        int q_offset = q_idx * EMBED_DIM + head * HEAD_DIM;
        int k_offset = k_idx * EMBED_DIM + head * HEAD_DIM;

        for (int d = 0; d < HEAD_DIM; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }

        int idx = head * SEQ_LEN * SEQ_LEN + q_idx * SEQ_LEN + k_idx;
        scores[idx] = score / sqrtf(HEAD_DIM);
    }
}

// Compute softmax per head
__global__ void softmax(float* scores, float* softmax_out) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y;

    if (head < NUM_HEADS && q_idx < SEQ_LEN) {
        int offset = head * SEQ_LEN * SEQ_LEN + q_idx * SEQ_LEN;

        float max_val = -1e9;
        for (int i = 0; i < SEQ_LEN; i++)
            max_val = fmaxf(max_val, scores[offset + i]);

        float sum = 0.0f;
        for (int i = 0; i < SEQ_LEN; i++)
            sum += expf(scores[offset + i] - max_val);

        for (int i = 0; i < SEQ_LEN; i++)
            softmax_out[offset + i] = expf(scores[offset + i] - max_val) / sum;
    }
}

// Compute final attention output
__global__ void attention_output(float* softmax_out, float* V, float* output) {
    int q_idx = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;

    if (q_idx < SEQ_LEN && head < NUM_HEADS && dim < HEAD_DIM) {
        float val = 0.0f;
        for (int k_idx = 0; k_idx < SEQ_LEN; k_idx++) {
            int softmax_idx = head * SEQ_LEN * SEQ_LEN + q_idx * SEQ_LEN + k_idx;
            int v_idx = k_idx * EMBED_DIM + head * HEAD_DIM + dim;
            val += softmax_out[softmax_idx] * V[v_idx];
        }

        int out_idx = q_idx * EMBED_DIM + head * HEAD_DIM + dim;
        output[out_idx] = val;
    }
}

int main() {
    size_t qkv_size = SEQ_LEN * EMBED_DIM * sizeof(float);
    size_t scores_size = NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float);

    // Allocate GPU memory
    float *Q, *K, *V, *scores, *softmax_out, *output;
    cudaMalloc(&Q, qkv_size);
    cudaMalloc(&K, qkv_size);
    cudaMalloc(&V, qkv_size);
    cudaMalloc(&scores, scores_size);
    cudaMalloc(&softmax_out, scores_size);
    cudaMalloc(&output, qkv_size);

    // Initialize host Q,K,V (dummy data)
    float host_Q[SEQ_LEN * EMBED_DIM];
    float host_K[SEQ_LEN * EMBED_DIM];
    float host_V[SEQ_LEN * EMBED_DIM];
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        host_Q[i] = 0.01f;
        host_K[i] = 0.02f;
        host_V[i] = 0.03f;
    }

    cudaMemcpy(Q, host_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, host_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, host_V, qkv_size, cudaMemcpyHostToDevice);

    // Compute attention scores
    dim3 grid_scores(NUM_HEADS, SEQ_LEN);
    attention_scores<<<grid_scores, SEQ_LEN>>>(Q, K, scores);
    cudaDeviceSynchronize();

    // Softmax normalization
    dim3 grid_softmax(NUM_HEADS, SEQ_LEN);
    softmax<<<grid_softmax, 1>>>(scores, softmax_out);
    cudaDeviceSynchronize();

    // Compute attention output
    dim3 grid_output(SEQ_LEN, NUM_HEADS);
    attention_output<<<grid_output, HEAD_DIM>>>(softmax_out, V, output);
    cudaDeviceSynchronize();

    // Verify (copy back some outputs)
    float host_output[SEQ_LEN * EMBED_DIM];
    cudaMemcpy(host_output, output, qkv_size, cudaMemcpyDeviceToHost);

    printf("Output [0,0]: %.4f\n", host_output[0]);
    printf("Output [0,1]: %.4f\n", host_output[1]);
    printf("Output [0,64]: %.4f\n", host_output[64]);  // Next head start

    // Cleanup
    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(scores); cudaFree(softmax_out); cudaFree(output);

    return 0;
}

