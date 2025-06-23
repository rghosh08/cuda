// paged_attention_config.h
#pragma once

#include <cstdint>

namespace paged_attention {

struct PagedAttentionConfig {
    int batch_size;         // Number of sequences in a batch
    int num_heads;          // Number of attention heads
    int head_dim;           // Dimension of each attention head
    int max_seq_len;        // Maximum sequence length
    int page_size;          // Number of tokens stored in each page
    int num_pages;          // Total number of pages in KV cache
    int omp_threads;        // Number of OpenMP threads to use
    int block_size;         // Tile size for FlashAttention
    bool recycle_pages;     // Whether to recycle pages from inactive sequences
};

// Default configuration
constexpr PagedAttentionConfig DEFAULT_CONFIG = {
    .batch_size = 32,
    .num_heads = 64,
    .head_dim = 128,
    .max_seq_len = 1024000,
    .page_size = 128,
    .num_pages = 2048,
    .omp_threads = 16,
    .block_size = 64,      // Default tile size for FlashAttention
    .recycle_pages = true  // Enable page recycling by default
};

} // namespace paged_attention
