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
    .batch_size = 2,
    .num_heads = 20,
    .head_dim = 8,
    .max_seq_len = 1048576,
    .page_size = 16,
    .num_pages = 32,
    .recycle_pages = true
};

} // namespace paged_attention
