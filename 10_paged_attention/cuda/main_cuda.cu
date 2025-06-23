// main_cuda.cu
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "paged_attention_config.h"
#include "paged_flash_attention_cuda.h"

// Function to detect system capabilities
void detect_system_capabilities() {
    std::cout << "=== Fixed Paged Attention CUDA Implementation ===" << std::endl;
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "System Information:" << std::endl;
    std::cout << "  CUDA Devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "  Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "    Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "    Total Global Memory: " << deviceProp.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "    Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "    Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "    Max Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    }
    
    std::cout << std::endl;
}

// Function to generate test data
void generate_test_data(
    int batch_size,
    int num_heads,
    int head_dim,
    std::vector<int>& seq_lengths,
    std::vector<float>& keys,
    std::vector<float>& values,
    std::vector<float>& queries
) {
    std::cout << "Generating test data with pattern: simple" << std::endl;
    
    // Generate sequence lengths
    seq_lengths.resize(batch_size);
    std::cout << "  Sequence lengths: ";
    for (int i = 0; i < batch_size; i++) {
        seq_lengths[i] = 16 + i * 4;  // 16, 20, 24, ..., etc.
        if (seq_lengths[i] > 120) seq_lengths[i] = 120;  // Cap at 120 for safety
        std::cout << seq_lengths[i] << " ";
    }
    std::cout << std::endl;
    
    // Calculate total tokens
    int total_tokens = 0;
    for (int len : seq_lengths) {
        total_tokens += len;
    }
    
    // Generate random data
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Resize and fill data
    keys.resize(total_tokens * num_heads * head_dim);
    values.resize(total_tokens * num_heads * head_dim);
    queries.resize(batch_size * num_heads * head_dim);
    
    for (size_t i = 0; i < keys.size(); i++) {
        keys[i] = dist(rng);
    }
    
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = dist(rng);
    }
    
    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = dist(rng);
    }
}

// Run the test
void run_test() {
    std::cout << "🧪 Running Simple Test 🧪" << std::endl << std::endl;
    
    // Create and initialize PagedFlashAttention
    paged_attention::PagedFlashAttentionCUDA attention;
    
    // Generate test data
    std::vector<int> seq_lengths;
    std::vector<float> keys, values, queries;
    generate_test_data(
        attention.get_config().batch_size,
        attention.get_config().num_heads,
        attention.get_config().head_dim,
        seq_lengths,
        keys,
        values,
        queries
    );
    
    // Update KV cache for each sequence
    int key_offset = 0;
    for (int seq_idx = 0; seq_idx < attention.get_config().batch_size; seq_idx++) {
        int seq_len = seq_lengths[seq_idx];
        
        if (!attention.update_kv_cache(
            seq_idx,
            seq_len,
            keys.data() + key_offset,
            values.data() + key_offset
        )) {
            std::cerr << "Failed to update KV cache for sequence " << seq_idx << std::endl;
            return;
        }
        
        key_offset += seq_len * attention.get_config().num_heads * attention.get_config().head_dim;
    }
    
    // Allocate output buffer
    std::vector<float> outputs(attention.get_config().batch_size * 
                             attention.get_config().num_heads * 
                             attention.get_config().head_dim, 0.0f);
    
    // Compute attention
    bool success = attention.compute_attention(
        queries.data(),
        outputs.data(),
        seq_lengths.data(),
        1.0f / std::sqrt(attention.get_config().head_dim)
    );
    
    if (success) {
        // Check for NaN/Inf in outputs
        bool valid_output = true;
        for (float val : outputs) {
            if (std::isnan(val) || std::isinf(val)) {
                valid_output = false;
                break;
            }
        }
        
        if (valid_output) {
            std::cout << "✅ Test completed successfully with valid outputs!" << std::endl;
            
            // Print some sample outputs
            std::cout << "Sample outputs:" << std::endl;
            for (int i = 0; i < std::min(5, attention.get_config().head_dim); i++) {
                std::cout << "  output[0][0][" << i << "] = " << outputs[i] << std::endl;
            }
        } else {
            std::cout << "❌ Test failed: Output contains NaN or Inf values" << std::endl;
        }
    } else {
        std::cout << "❌ Test failed: Attention computation failed" << std::endl;
    }
}

// Main function
int main() {
    // Detect system capabilities
    detect_system_capabilities();
    
    // Run the test
    run_test();
    
    return 0;
}
