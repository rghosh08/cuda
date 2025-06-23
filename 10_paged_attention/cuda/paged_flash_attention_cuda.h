// paged_flash_attention_cuda.h
#pragma once

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "paged_attention_config.h"

namespace paged_attention {

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } \
}

class PagedFlashAttentionCUDA {
private:
    // Configuration
    PagedAttentionConfig config;
    
    // GPU Memory
    float* d_key_cache;
    float* d_value_cache;
    int* d_page_table;
    
    // Host memory for page management
    std::vector<std::vector<int>> seq_to_page_mapping;
    std::vector<int> free_page_indices;
    
    // Memory management
    bool allocate_device_memory();
    void free_device_memory();
    bool allocate_pages_for_sequence(int seq_idx, int num_tokens);
    
public:
    // Constructor/Destructor
    PagedFlashAttentionCUDA(const PagedAttentionConfig& config = DEFAULT_CONFIG);
    ~PagedFlashAttentionCUDA();
    
    // Initialize with configuration
    bool initialize(const PagedAttentionConfig& new_config);
    
    // Update KV cache for a sequence
    bool update_kv_cache(
        int seq_idx,
        int seq_len,
        const float* h_keys,
        const float* h_values
    );
    
    // Compute attention using a simple approach
    bool compute_attention(
        const float* h_queries,
        float* h_outputs,
        const int* h_seq_lengths,
        float scale = 1.0f
    );
    
    // Get current configuration
    const PagedAttentionConfig& get_config() const {
        return config;
    }
    
    // Debug helpers
    void print_memory_usage() const;
};

} // namespace paged_attention
