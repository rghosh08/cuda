// paged_attention_config.h
#pragma once

namespace paged_attention {

struct PagedAttentionConfig {
    int batch_size;
    int num_heads;
    int head_dim;
    int max_seq_len;
    int page_size;
    int num_pages;
    bool recycle_pages;
};

// Minimal configuration that will definitely work
constexpr PagedAttentionConfig DEFAULT_CONFIG = {
    .batch_size = 64, // [ 2, 4, 8, 16, 32, 64]
    .num_heads = 32,
    .head_dim = 64,
    .max_seq_len = 512, // [512, 1024, 2048, 4096, 8192]
    .page_size = 16,
    .num_pages = 2048, 
    .recycle_pages = true
};

} // namespace paged_attention
