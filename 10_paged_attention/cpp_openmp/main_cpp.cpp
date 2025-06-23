// main.cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>
#include <omp.h>

#include "paged_attention_config.h"
#include "paged_flash_attention.h"

// Function to detect system capabilities
void detect_system_capabilities() {
    std::cout << "=== Fixed Paged Attention C++ OpenMP Implementation ===" << std::endl;
    std::cout << "System Information:" << std::endl;
    std::cout << "  Max OpenMP threads: " << omp_get_max_threads() << std::endl;
    std::cout << "  CPU cache line size: 64 bytes" << std::endl;
    std::cout << "  OpenMP version: " << _OPENMP << std::endl;
    
    // Check for AVX2 support
    #ifdef __AVX2__
    std::cout << "  AVX2 support: Available" << std::endl;
    #else
    std::cout << "  AVX2 support: Not available" << std::endl;
    #endif
    
    std::cout << std::endl;
}

// Function to generate test data
void generate_test_data(
    int batch_size,
    int max_seq_len,
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

// Main function to run tests
int main() {
    // Detect system capabilities
    detect_system_capabilities();
    
    // Run simple test
    std::cout << "🧪 Running Simple Test 🧪" << std::endl << std::endl;
    
    // Configure PagedFlashAttention
    paged_attention::PagedAttentionConfig config = paged_attention::DEFAULT_CONFIG;
    
    // Create and initialize PagedFlashAttention
    paged_attention::PagedFlashAttention attention(config);
    
    // Generate test data
    std::vector<int> seq_lengths;
    std::vector<float> keys, values, queries;
    generate_test_data(
        config.batch_size,
        config.max_seq_len,
        config.num_heads,
        config.head_dim,
        seq_lengths,
        keys,
        values,
        queries
    );
    
    // Update KV cache for each sequence
    int key_offset = 0;
    for (int seq_idx = 0; seq_idx < config.batch_size; seq_idx++) {
        int seq_len = seq_lengths[seq_idx];
        
        // Update KV cache
        if (!attention.update_kv_cache(
            seq_idx,
            seq_len,
            keys.data() + key_offset,
            values.data() + key_offset
        )) {
            std::cerr << "Failed to update KV cache for sequence " << seq_idx << std::endl;
            return 1;
        }
        
        // Update offset
        key_offset += seq_len * config.num_heads * config.head_dim;
    }
    
    // Allocate output buffer
    std::vector<float> outputs(config.batch_size * config.num_heads * config.head_dim, 0.0f);
    
    // Compute attention
    attention.compute_attention(
        queries.data(),
        outputs.data(),
        seq_lengths.data(),
        1.0f / std::sqrt(config.head_dim)
    );
    
    std::cout << "✅ PagedFlashAttention test completed successfully!" << std::endl;
    
    return 0;
}
