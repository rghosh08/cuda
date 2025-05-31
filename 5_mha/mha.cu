#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define SEQ_LEN 64
#define EMBED_DIM 256
#define NUM_HEADS 8
#define HEAD_DIM (EMBED_DIM / NUM_HEADS)

// Kernel: Compute scaled dot-product attention scores
__global__ void attention_scores(const float* Q, const float* K, float* scores, int seq_len) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y;
    int k_idx = threadIdx.x;

    if (k_idx < seq_len) {
        float score = 0.0f;
        int q_offset = q_idx * EMBED_DIM + head * HEAD_DIM;
        int k_offset = k_idx * EMBED_DIM + head * HEAD_DIM;

        for (int d = 0; d < HEAD_DIM; ++d)
            score += Q[q_offset + d] * K[k_offset + d];

        int idx = head * seq_len * seq_len + q_idx * seq_len + k_idx;
        scores[idx] = score / sqrtf(HEAD_DIM);
    }
}

// Optimized parallel softmax kernel
__global__ void softmax(float* scores, float* softmax_out, int seq_len) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    int offset = head * seq_len * seq_len + q_idx * seq_len;

    // Find max value
    float max_val = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x)
        max_val = fmaxf(max_val, scores[offset + i]);

    sdata[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute sum
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x)
        sum += expf(scores[offset + i] - max_val);

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // Final softmax output
    for (int i = tid; i < seq_len; i += blockDim.x)
        softmax_out[offset + i] = expf(scores[offset + i] - max_val) / sum;
}

// Compute attention output
__global__ void attention_output(const float* softmax_out, const float* V, float* output, int seq_len) {
    int q_idx = blockIdx.x;
    int head = blockIdx.y;
    int dim = threadIdx.x;

    if (dim < HEAD_DIM) {
        float val = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            int softmax_idx = head * seq_len * seq_len + q_idx * seq_len + k_idx;
            int v_idx = k_idx * EMBED_DIM + head * HEAD_DIM + dim;
            val += softmax_out[softmax_idx] * V[v_idx];
        }

        int out_idx = q_idx * EMBED_DIM + head * HEAD_DIM + dim;
        output[out_idx] = val;
    }
}

int main() {
    size_t tensor_size = SEQ_LEN * EMBED_DIM * sizeof(float);
    size_t scores_size = NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float);

    float *Q, *K, *V, *scores, *softmax_out, *output;
    cudaMalloc(&Q, tensor_size);
    cudaMalloc(&K, tensor_size);
    cudaMalloc(&V, tensor_size);
    cudaMalloc(&scores, scores_size);
    cudaMalloc(&softmax_out, scores_size);
    cudaMalloc(&output, tensor_size);

    float h_Q[SEQ_LEN * EMBED_DIM], h_K[SEQ_LEN * EMBED_DIM], h_V[SEQ_LEN * EMBED_DIM];
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        h_Q[i] = 0.01f; h_K[i] = 0.02f; h_V[i] = 0.03f;
    }

    cudaMemcpy(Q, h_Q, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, h_K, tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, h_V, tensor_size, cudaMemcpyHostToDevice);

    dim3 grid_scores(NUM_HEADS, SEQ_LEN);
    attention_scores<<<grid_scores, SEQ_LEN>>>(Q, K, scores, SEQ_LEN);

    dim3 grid_softmax(NUM_HEADS, SEQ_LEN);
    softmax<<<grid_softmax, 128, 128 * sizeof(float)>>>(scores, softmax_out, SEQ_LEN);

    dim3 grid_output(SEQ_LEN, NUM_HEADS);
    attention_output<<<grid_output, HEAD_DIM>>>(softmax_out, V, output, SEQ_LEN);

    cudaDeviceSynchronize();

    float h_output[SEQ_LEN * EMBED_DIM];
    cudaMemcpy(h_output, output, tensor_size, cudaMemcpyDeviceToHost);

    printf("Output[0]: %.4f\n", h_output[0]);
    printf("Output[1]: %.4f\n", h_output[1]);
    printf("Output[64]: %.4f\n", h_output[64]);

    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(scores); cudaFree(softmax_out); cudaFree(output);

    return 0;
}
