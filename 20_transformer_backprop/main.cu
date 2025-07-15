#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Utility functions
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Softmax backward kernel
__global__ void softmax_backward_kernel(
    const float* grad_output,
    const float* softmax_out,
    float* grad_input,
    int batch_size,
    int seq_len
) {
    int batch_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int col_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || row_idx >= seq_len || col_idx >= seq_len) return;
    
    int base_idx = batch_idx * seq_len * seq_len + row_idx * seq_len;
    
    // Compute sum for this row
    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        sum += grad_output[base_idx + k] * softmax_out[base_idx + k];
    }
    
    // Compute gradient
    float s_i = softmax_out[base_idx + col_idx];
    float grad_out_i = grad_output[base_idx + col_idx];
    
    grad_input[base_idx + col_idx] = s_i * (grad_out_i - sum);
}

// Layer normalization backward kernel
__global__ void layernorm_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx >= total_elements) return;
    
    int batch_seq_idx = idx / hidden_size;
    int hidden_idx = idx % hidden_size;
    
    // Compute mean and variance for this batch/sequence position
    float mean = 0.0f, var = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        float val = input[batch_seq_idx * hidden_size + h];
        mean += val;
    }
    mean /= hidden_size;
    
    for (int h = 0; h < hidden_size; h++) {
        float val = input[batch_seq_idx * hidden_size + h];
        var += (val - mean) * (val - mean);
    }
    var /= hidden_size;
    
    float std_inv = rsqrtf(var + eps);
    float x_norm = (input[idx] - mean) * std_inv;
    
    // Accumulate gradients for gamma and beta
    atomicAdd(&grad_gamma[hidden_idx], grad_output[idx] * x_norm);
    atomicAdd(&grad_beta[hidden_idx], grad_output[idx]);
    
    // Compute gradient w.r.t. input
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        int h_idx = batch_seq_idx * hidden_size + h;
        sum1 += grad_output[h_idx];
        sum2 += grad_output[h_idx] * (input[h_idx] - mean);
    }
    
    float grad_input_val = std_inv * (grad_output[idx] - sum1 / hidden_size - 
                                     (input[idx] - mean) * sum2 / (hidden_size * (var + eps)));
    grad_input[idx] = gamma[hidden_idx] * grad_input_val;
}

// Multi-head attention backward kernel
__global__ void attention_backward_kernel(
    const float* grad_output,
    const float* query,
    const float* key,
    const float* value,
    const float* attention_weights,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / (seq_len * hidden_size);
    int seq_idx = (idx / hidden_size) % seq_len;
    int hidden_idx = idx % hidden_size;
    
    int head_idx = hidden_idx / (hidden_size / num_heads);
    
    // Simplified attention backward - in practice, you'd need proper attention computation
    float grad_q = 0.0f, grad_k = 0.0f, grad_v = 0.0f;
    
    for (int j = 0; j < seq_len; j++) {
        int attn_idx = batch_idx * num_heads * seq_len * seq_len + 
                       head_idx * seq_len * seq_len + seq_idx * seq_len + j;
        float attn_weight = attention_weights[attn_idx];
        
        int j_idx = batch_idx * seq_len * hidden_size + j * hidden_size + hidden_idx;
        
        // Gradient w.r.t. value
        grad_v += attn_weight * grad_output[idx];
        
        // Gradient w.r.t. query and key (simplified)
        grad_q += scale * key[j_idx] * grad_output[idx];
        grad_k += scale * query[idx] * grad_output[idx];
    }
    
    atomicAdd(&grad_query[idx], grad_q);
    atomicAdd(&grad_key[idx], grad_k);
    atomicAdd(&grad_value[idx], grad_v);
}

// GELU backward kernel
__global__ void gelu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    float x = input[idx];
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
    float tanh_out = tanhf(tanh_arg);
    float cdf = 0.5f * (1.0f + tanh_out);
    
    float tanh_derivative = 1.0f - tanh_out * tanh_out;
    float pdf_factor = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    
    float gelu_grad = cdf + x * 0.5f * tanh_derivative * pdf_factor;
    grad_input[idx] = grad_output[idx] * gelu_grad;
}

// Main transformer backward function
__host__ void transformer_layer_backward(
    const float* grad_output,
    const float* input,
    const float* query,
    const float* key,
    const float* value,
    const float* attention_weights,
    const float* ffn_input,
    const float* gamma1,
    const float* gamma2,
    float* grad_input,
    float* grad_query,
    float* grad_key,
    float* grad_value,
    float* grad_ffn,
    float* grad_gamma1,
    float* grad_beta1,
    float* grad_gamma2,
    float* grad_beta2,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_heads,
    cudaStream_t stream = 0
) {
    int total_elements = batch_size * seq_len * hidden_size;
    
    // Configure kernel launch parameters
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // 1. Backward through second layer norm
    layernorm_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, input, gamma2, grad_input, grad_gamma2, grad_beta2,
        batch_size, seq_len, hidden_size, 1e-5f
    );
    
    // 2. Backward through GELU activation
    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_input, ffn_input, grad_ffn, total_elements
    );
    
    // 3. Backward through first layer norm
    layernorm_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_ffn, input, gamma1, grad_input, grad_gamma1, grad_beta1,
        batch_size, seq_len, hidden_size, 1e-5f
    );
    
    // 4. Backward through attention
    attention_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_input, query, key, value, attention_weights,
        grad_query, grad_key, grad_value,
        batch_size, seq_len, hidden_size, num_heads, 1.0f / sqrtf(hidden_size / num_heads)
    );
    
    // 5. Backward through softmax (for attention weights)
    dim3 softmax_grid(batch_size, seq_len);
    dim3 softmax_block(min(seq_len, 1024));
    
    // Note: This would need proper gradient flow from attention backward
    // softmax_backward_kernel<<<softmax_grid, softmax_block, 0, stream>>>(
    //     attention_grad, attention_weights, attention_scores_grad,
    //     batch_size, seq_len
    // );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Test function
void test_transformer_backward() {
    // Model parameters
    const int batch_size = 2;
    const int seq_len = 64;
    const int hidden_size = 512;
    const int num_heads = 8;
    
    const int total_elements = batch_size * seq_len * hidden_size;
    const int attn_elements = batch_size * num_heads * seq_len * seq_len;
    
    // Allocate host memory
    std::vector<float> h_grad_output(total_elements, 0.1f);
    std::vector<float> h_input(total_elements);
    std::vector<float> h_query(total_elements);
    std::vector<float> h_key(total_elements);
    std::vector<float> h_value(total_elements);
    std::vector<float> h_attention_weights(attn_elements);
    std::vector<float> h_ffn_input(total_elements);
    std::vector<float> h_gamma1(hidden_size, 1.0f);
    std::vector<float> h_gamma2(hidden_size, 1.0f);
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (int i = 0; i < total_elements; i++) {
        h_input[i] = dist(gen);
        h_query[i] = dist(gen);
        h_key[i] = dist(gen);
        h_value[i] = dist(gen);
        h_ffn_input[i] = dist(gen);
    }
    
    for (int i = 0; i < attn_elements; i++) {
        h_attention_weights[i] = 1.0f / seq_len; // Uniform attention
    }
    
    // Allocate device memory
    float *d_grad_output, *d_input, *d_query, *d_key, *d_value;
    float *d_attention_weights, *d_ffn_input, *d_gamma1, *d_gamma2;
    float *d_grad_input, *d_grad_query, *d_grad_key, *d_grad_value, *d_grad_ffn;
    float *d_grad_gamma1, *d_grad_beta1, *d_grad_gamma2, *d_grad_beta2;
    
    CUDA_CHECK(cudaMalloc(&d_grad_output, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_key, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_value, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attention_weights, attn_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_input, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2, hidden_size * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_query, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_key, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_value, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_ffn, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma2, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta2, hidden_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attention_weights, h_attention_weights.data(), attn_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ffn_input, h_ffn_input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma1, h_gamma1.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma2, h_gamma2.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Zero gradients
    CUDA_CHECK(cudaMemset(d_grad_input, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_query, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_key, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_value, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_ffn, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_gamma1, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta1, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_gamma2, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta2, 0, hidden_size * sizeof(float)));
    
    // Run backward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    transformer_layer_backward(
        d_grad_output, d_input, d_query, d_key, d_value,
        d_attention_weights, d_ffn_input, d_gamma1, d_gamma2,
        d_grad_input, d_grad_query, d_grad_key, d_grad_value, d_grad_ffn,
        d_grad_gamma1, d_grad_beta1, d_grad_gamma2, d_grad_beta2,
        batch_size, seq_len, hidden_size, num_heads
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back
    std::vector<float> h_grad_input(total_elements);
    CUDA_CHECK(cudaMemcpy(h_grad_input.data(), d_grad_input, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute gradient norm
    float grad_norm = 0.0f;
    for (float g : h_grad_input) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);
    
    printf("Transformer backward pass completed!\n");
    printf("Execution time: %ld microseconds\n", duration.count());
    printf("Input gradient L2 norm: %f\n", grad_norm);
    printf("Sample gradients: %f, %f, %f\n", h_grad_input[0], h_grad_input[1], h_grad_input[2]);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_attention_weights));
    CUDA_CHECK(cudaFree(d_ffn_input));
    CUDA_CHECK(cudaFree(d_gamma1));
    CUDA_CHECK(cudaFree(d_gamma2));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_query));
    CUDA_CHECK(cudaFree(d_grad_key));
    CUDA_CHECK(cudaFree(d_grad_value));
    CUDA_CHECK(cudaFree(d_grad_ffn));
    CUDA_CHECK(cudaFree(d_grad_gamma1));
    CUDA_CHECK(cudaFree(d_grad_beta1));
    CUDA_CHECK(cudaFree(d_grad_gamma2));
    CUDA_CHECK(cudaFree(d_grad_beta2));
}

int main() {
    printf("Running standalone transformer backward CUDA function...\n");
    test_transformer_backward();
    printf("Test completed successfully!\n");
    return 0;
}

// Compile with: nvcc -o standalone_transformer standalone_transformer.cu -lcublas