// paged_flash_attention.h
#pragma once

#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include "paged_attention_config.h"

namespace paged_attention {

class PagedFlashAttention {
private:
    // Configuration
    PagedAttentionConfig config;
    
    // Memory for key-value cache
    float* key_cache;
    float* value_cache;
    
    // Page table: maps (batch_idx, token_idx) to physical page locations
    int* page_table;
    
    // Attention weights buffer
    float* attn_weights;
    
    // Track allocated pages per sequence
    std::vector<std::vector<int>> seq_to_page_mapping;
    
    // Free page tracking
    std::vector<int> free_page_indices;
    
    // Memory management helpers
    bool allocate_memory();
    void free_memory();
    
    // Page allocation and recycling
    bool allocate_pages_for_sequence(int seq_idx, int num_tokens);
    void reclaim_pages_from_inactive_sequences();
    
public:
    // Constructor/Destructor
    PagedFlashAttention(const PagedAttentionConfig& config = DEFAULT_CONFIG);
    ~PagedFlashAttention();
    
    // Initialize with configuration
    bool initialize(const PagedAttentionConfig& new_config);
    
    // Update KV cache for a sequence
    bool update_kv_cache(
        int seq_idx,
        int seq_len,
        float* keys,    // [seq_len, num_heads, head_dim]
        float* values   // [seq_len, num_heads, head_dim]
    );
    
    // Compute attention using FlashAttention algorithm
    bool compute_attention(
        float* queries,      // [batch_size, num_heads, head_dim]
        float* outputs,      // [batch_size, num_heads, head_dim]
        int* seq_lengths,    // [batch_size]
        float scale = 1.0f   // Scaling factor for attention scores
    );
    
    // Debug helpers
    void print_memory_usage() const;
    void print_page_allocation(int seq_idx) const;
};

} // namespace paged_attention
