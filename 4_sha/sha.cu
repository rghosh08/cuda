#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SEQ_LEN 128
#define HEAD_DIM 64

// Kernel: Compute attention scores = Q x K^T
__global__ void attention_scores(float* Q, float* K, float* scores) {
    int q_idx = blockIdx.x;
    int k_idx = threadIdx.x;

    if (q_idx < SEQ_LEN && k_idx < SEQ_LEN) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += Q[q_idx * HEAD_DIM + d] * K[k_idx * HEAD_DIM + d];
        }
        scores[q_idx * SEQ_LEN + k_idx] = score / sqrtf(HEAD_DIM);
    }
}

// Kernel: Softmax over each row
__global__ void softmax(float* scores, float* softmax_out) {
    int q_idx = blockIdx.x;

    if (q_idx < SEQ_LEN) {
        float max_score = -1e9;
        for (int i = 0; i < SEQ_LEN; i++) {
            float val = scores[q_idx * SEQ_LEN + i];
            if (val > max_score) max_score = val;
        }

        // Compute denominator
        float sum = 0.0f;
        for (int i = 0; i < SEQ_LEN; i++) {
            sum += expf(scores[q_idx * SEQ_LEN + i] - max_score);
        }

        // Normalize scores
        for (int i = 0; i < SEQ_LEN; i++) {
            softmax_out[q_idx * SEQ_LEN + i] = expf(scores[q_idx * SEQ_LEN + i] - max_score) / sum;
        }
    }
}

// Kernel: Multiply softmax_out with V to get attention output
__global__ void compute_attention_output(float* softmax_out, float* V, float* output) {
    int q_idx = blockIdx.x;
    int dim = threadIdx.x;

    if (q_idx < SEQ_LEN && dim < HEAD_DIM) {
        float val = 0.0f;
        for (int k_idx = 0; k_idx < SEQ_LEN; k_idx++) {
            val += softmax_out[q_idx * SEQ_LEN + k_idx] * V[k_idx * HEAD_DIM + dim];
        }
        output[q_idx * HEAD_DIM + dim] = val;
    }
}

int main() {
    size_t qkv_size = SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t scores_size = SEQ_LEN * SEQ_LEN * sizeof(float);

    // Allocate GPU memory
    float *Q, *K, *V, *scores, *softmax_out, *output;
    cudaMalloc(&Q, qkv_size);
    cudaMalloc(&K, qkv_size);
    cudaMalloc(&V, qkv_size);
    cudaMalloc(&scores, scores_size);
    cudaMalloc(&softmax_out, scores_size);
    cudaMalloc(&output, qkv_size);

    // Initialize host data (example)
    float host_Q[SEQ_LEN * HEAD_DIM], host_K[SEQ_LEN * HEAD_DIM], host_V[SEQ_LEN * HEAD_DIM];
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) {
        host_Q[i] = 0.01f; host_K[i] = 0.02f; host_V[i] = 0.03f;
    }

    // Copy to GPU
    cudaMemcpy(Q, host_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, host_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, host_V, qkv_size, cudaMemcpyHostToDevice);

    // Compute attention scores (Q*K^T)
    attention_scores<<<SEQ_LEN, SEQ_LEN>>>(Q, K, scores);
    cudaDeviceSynchronize();

    // Softmax normalization
    softmax<<<SEQ_LEN, 1>>>(scores, softmax_out);
    cudaDeviceSynchronize();

    // Compute final attention output (softmax_out*V)
    compute_attention_output<<<SEQ_LEN, HEAD_DIM>>>(softmax_out, V, output);
    cudaDeviceSynchronize();

    // Copy results back to verify
    float host_output[SEQ_LEN * HEAD_DIM];
    cudaMemcpy(host_output, output, qkv_size, cudaMemcpyDeviceToHost);

    // Print some output values for verification
    printf("Attention output at [0,0]: %.4f\n", host_output[0]);
    printf("Attention output at [0,1]: %.4f\n", host_output[1]);
    printf("Attention output at [1,0]: %.4f\n", host_output[HEAD_DIM]);

    // Cleanup
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(scores);
    cudaFree(softmax_out);
    cudaFree(output);

    return 0;
}

