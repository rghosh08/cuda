// paged_flash_attention.cpp
#include "paged_flash_attention.h"
#include <algorithm>
#include <limits>

namespace paged_attention {

PagedFlashAttention::PagedFlashAttention(const PagedAttentionConfig& config)
    : config(config),
      key_cache(nullptr),
      value_cache(nullptr),
      page_table(nullptr),
      attn_weights(nullptr) {
    
    // Initialize with provided configuration
    initialize(config);
}

PagedFlashAttention::~PagedFlashAttention() {
    free_memory();
}

bool PagedFlashAttention::initialize(const PagedAttentionConfig& new_config) {
    // Update configuration
    config = new_config;
    
    // Set OpenMP threads
    omp_set_num_threads(config.omp_threads);
    
    // Free existing memory if any
    free_memory();
    
    // Initialize sequence to page mapping
    seq_to_page_mapping.resize(config.batch_size);
    for (auto& mapping : seq_to_page_mapping) {
        mapping.clear();
    }
    
    // Initialize free page list
    free_page_indices.clear();
    for (int i = 0; i < config.num_pages; i++) {
        free_page_indices.push_back(i);
    }
    
    // Allocate memory
    if (!allocate_memory()) {
        std::cerr << "Failed to allocate memory for PagedFlashAttention" << std::endl;
        return false;
    }
    
    // Print memory usage info
    print_memory_usage();
    
    std::cout << "✅ PagedFlashAttention OpenMP initialized successfully!" << std::endl;
    return true;
}

bool PagedFlashAttention::allocate_memory() {
    try {
        // Calculate memory sizes
        size_t kv_cache_size = static_cast<size_t>(config.num_pages) * 
                              config.page_size * 
                              config.num_heads * 
                              config.head_dim;
        
        size_t page_table_size = static_cast<size_t>(config.batch_size) * 
                                config.max_seq_len;
        
        // Allocate key-value cache
        key_cache = new float[kv_cache_size];
        value_cache = new float[kv_cache_size];
        
        // Allocate page table
        page_table = new int[page_table_size];
        
        // Initialize page table with -1 (no page allocated)
        std::fill(page_table, page_table + page_table_size, -1);
        
        // Allocate attention weights buffer if needed
        size_t attn_weights_size = static_cast<size_t>(config.batch_size) * 
                                  config.num_heads * 
                                  config.max_seq_len;
        attn_weights = new float[attn_weights_size];
        
        return true;
    } catch (const std::bad_alloc&) {
        std::cerr << "Memory allocation failed" << std::endl;
        free_memory();
        return false;
    }
}

void PagedFlashAttention::free_memory() {
    delete[] key_cache;
    delete[] value_cache;
    delete[] page_table;
    delete[] attn_weights;
    
    key_cache = nullptr;
    value_cache = nullptr;
    page_table = nullptr;
    attn_weights = nullptr;
}

void PagedFlashAttention::print_memory_usage() const {
    // Calculate memory usage in bytes
    size_t kv_cache_bytes = 2 * static_cast<size_t>(config.num_pages) * 
                           config.page_size * 
                           config.num_heads * 
                           config.head_dim * 
                           sizeof(float);
    
    size_t page_table_bytes = static_cast<size_t>(config.batch_size) * 
                             config.max_seq_len * 
                             sizeof(int);
    
    size_t attn_weights_bytes = static_cast<size_t>(config.batch_size) * 
                               config.num_heads * 
                               config.max_seq_len * 
                               sizeof(float);
    
    // Convert to KB/MB for display
    double kv_cache_mb = kv_cache_bytes / (1024.0 * 1024.0);
    double page_table_kb = page_table_bytes / 1024.0;
    double attn_weights_kb = attn_weights_bytes / 1024.0;
    
    std::cout << "Memory allocation:" << std::endl;
    std::cout << "  Key/Value cache: " << kv_cache_mb << " MB" << std::endl;
    std::cout << "  Page table: " << page_table_kb << " KB" << std::endl;
    std::cout << "  Attention weights: " << attn_weights_kb << " KB" << std::endl;
}

bool PagedFlashAttention::allocate_pages_for_sequence(int seq_idx, int num_tokens) {
    if (seq_idx < 0 || seq_idx >= config.batch_size) {
        std::cerr << "Invalid sequence index: " << seq_idx << std::endl;
        return false;
    }
    
    // Calculate number of pages needed
    int pages_needed = (num_tokens + config.page_size - 1) / config.page_size;
    
    // Check if we have enough free pages
    if (free_page_indices.size() < pages_needed) {
        if (config.recycle_pages) {
            // Try to reclaim pages from inactive sequences
            reclaim_pages_from_inactive_sequences();
        }
        
        // Check again after attempted recycling
        if (free_page_indices.size() < pages_needed) {
            std::cerr << "❌ Test failed: Out of pages: needed " << pages_needed 
                      << ", got " << free_page_indices.size() << std::endl;
            return false;
        }
    }
    
    // Allocate pages to this sequence
    std::vector<int> allocated_pages;
    for (int i = 0; i < pages_needed; i++) {
        int page_idx = free_page_indices.back();
        free_page_indices.pop_back();
        allocated_pages.push_back(page_idx);
    }
    
    // Store allocated pages for this sequence
    seq_to_page_mapping[seq_idx] = allocated_pages;
    
    // Print allocation info
    std::cout << "  Allocated " << pages_needed << " pages for sequence " << seq_idx << ": ";
    for (int page : allocated_pages) {
        std::cout << page << " ";
    }
    std::cout << std::endl;
    
    return true;
}

void PagedFlashAttention::reclaim_pages_from_inactive_sequences() {
    // This is a simple implementation that could be improved with LRU policy
    // In a real system, you would track sequence activity and reclaim from inactive ones
    
    for (int seq_idx = 0; seq_idx < config.batch_size; seq_idx++) {
        auto& pages = seq_to_page_mapping[seq_idx];
        if (!pages.empty()) {
            // Add all pages from this sequence back to free list
            free_page_indices.insert(free_page_indices.end(), pages.begin(), pages.end());
            pages.clear();
            
            // Update page table to indicate no pages allocated
            for (int token_idx = 0; token_idx < config.max_seq_len; token_idx++) {
                page_table[seq_idx * config.max_seq_len + token_idx] = -1;
            }
            
            std::cout << "  Reclaimed pages from sequence " << seq_idx << std::endl;
            
            // Break if we've reclaimed enough pages
            if (free_page_indices.size() >= config.num_pages / 2) {
                break;
            }
        }
    }
}

bool PagedFlashAttention::update_kv_cache(
    int seq_idx, int seq_len, float* keys, float* values) {
    
    std::cout << "Updating KV cache for sequence " << seq_idx << " (length: " << seq_len << ")" << std::endl;
    
    // Allocate pages for this sequence
    if (!allocate_pages_for_sequence(seq_idx, seq_len)) {
        return false;
    }
    
    // Get allocated pages for this sequence
    const auto& allocated_pages = seq_to_page_mapping[seq_idx];
    
    // Update page table and copy data to KV cache
    int tokens_updated = 0;
    int current_page_idx = 0;
    int page_offset = 0;
    
    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        // If we've filled the current page, move to the next one
        if (page_offset >= config.page_size) {
            current_page_idx++;
            page_offset = 0;
        }
        
        // Get physical page index
        int physical_page = allocated_pages[current_page_idx];
        
        // Update page table
        page_table[seq_idx * config.max_seq_len + token_idx] = physical_page;
        
        // Copy key and value data to cache
        size_t cache_offset = (physical_page * config.page_size + page_offset) * 
                              config.num_heads * config.head_dim;
        
        size_t input_offset = token_idx * config.num_heads * config.head_dim;
        
        // Copy keys
        std::memcpy(
            key_cache + cache_offset,
            keys + input_offset,
            config.num_heads * config.head_dim * sizeof(float)
        );
        
        // Copy values
        std::memcpy(
            value_cache + cache_offset,
            values + input_offset,
            config.num_heads * config.head_dim * sizeof(float)
        );
        
        // Update counters
        tokens_updated++;
        page_offset++;
    }
    
    std::cout << "  Updated " << allocated_pages.size() << " pages with " 
              << tokens_updated << " tokens" << std::endl;
    
    return true;
}

bool PagedFlashAttention::compute_attention(
    float* queries, float* outputs, int* seq_lengths, float scale) {
    
    // Process all sequences in parallel
    #pragma omp parallel for
    for (int b = 0; b < config.batch_size; b++) {
        // Skip if no sequence length is provided
        if (seq_lengths[b] <= 0) continue;
        
        // Process each head
        for (int h = 0; h < config.num_heads; h++) {
            // Get query pointer for this batch and head
            float* q_ptr = queries + b * config.num_heads * config.head_dim + h * config.head_dim;
            
            // Get output pointer for this batch and head
            float* o_ptr = outputs + b * config.num_heads * config.head_dim + h * config.head_dim;
            
            // Initialize output to zeros
            std::memset(o_ptr, 0, config.head_dim * sizeof(float));
            
            // Get sequence length for this batch
            int seq_len = seq_lengths[b];
            
            // FlashAttention algorithm variables
            float m = -std::numeric_limits<float>::infinity();  // Max for numerical stability
            float l = 0.0f;  // Sum of exp values
            
            // Process in tiles to maximize cache efficiency
            for (int start_idx = 0; start_idx < seq_len; start_idx += config.block_size) {
                int end_idx = std::min(start_idx + config.block_size, seq_len);
                int block_len = end_idx - start_idx;
                
                // Allocate temporary memory for this block's scores
                float* block_scores = static_cast<float*>(alloca(block_len * sizeof(float)));
                
                // Compute attention scores for this block
                float block_max = -std::numeric_limits<float>::infinity();
                
                // Compute attention scores for each token in the block
                for (int i = 0; i < block_len; i++) {
                    int token_idx = start_idx + i;
                    
                    // Get page and offset for this token
                    int page_idx = page_table[b * config.max_seq_len + token_idx];
                    if (page_idx < 0) {
                        // This token doesn't have a page allocated, skip it
                        block_scores[i] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    
                    int offset = token_idx % config.page_size;
                    
                    // Get key pointer from cache
                    float* k_ptr = key_cache + 
                        (page_idx * config.page_size + offset) * config.num_heads * config.head_dim + 
                        h * config.head_dim;
                    
                    // Compute attention score (dot product)
                    float score = 0.0f;
                    for (int d = 0; d < config.head_dim; d++) {
                        score += q_ptr[d] * k_ptr[d];
                    }
                    score *= scale;
                    
                    // Store score
                    block_scores[i] = score;
                    block_max = std::max(block_max, score);
                }
                
                // Update max value for numerical stability
                float m_old = m;
                m = std::max(m, block_max);
                
                // Skip if this block has no valid scores
                if (m == -std::numeric_limits<float>::infinity()) {
                    continue;
                }
                
                // Scale factor for re-normalization
                float m_diff = std::exp(m_old - m);
                
                // Scale output by exp(m_old - m) if not the first block
                if (start_idx > 0 && m_old != -std::numeric_limits<float>::infinity()) {
                    for (int d = 0; d < config.head_dim; d++) {
                        o_ptr[d] *= m_diff;
                    }
                    l *= m_diff;
                }
                
                // Compute softmax and weighted sum for this block
                for (int i = 0; i < block_len; i++) {
                    int token_idx = start_idx + i;
                    
                    // Skip if score is -infinity (invalid token)
                    if (block_scores[i] == -std::numeric_limits<float>::infinity()) {
                        continue;
                    }
                    
                    // Compute exp(score - max) for numerical stability
                    float exp_score = std::exp(block_scores[i] - m);
                    
                    // Get page and offset for this token
                    int page_idx = page_table[b * config.max_seq_len + token_idx];
                    int offset = token_idx % config.page_size;
                    
                    // Get value pointer from cache
                    float* v_ptr = value_cache + 
                        (page_idx * config.page_size + offset) * config.num_heads * config.head_dim + 
                        h * config.head_dim;
                    
                    // Accumulate weighted value
                    for (int d = 0; d < config.head_dim; d++) {
                        o_ptr[d] += exp_score * v_ptr[d];
                    }
                    
                    // Accumulate normalizing factor
                    l += exp_score;
                }
            }
            
            // Skip normalization if l is zero (no valid tokens)
            if (l == 0.0f) {
                continue;
            }
            
            // Normalize output by denominator (sum of all exponential terms)
            for (int d = 0; d < config.head_dim; d++) {
                o_ptr[d] /= l;
            }
        }
    }
    
    return true;
}

void PagedFlashAttention::print_page_allocation(int seq_idx) const {
    if (seq_idx < 0 || seq_idx >= config.batch_size || seq_to_page_mapping[seq_idx].empty()) {
        std::cout << "No pages allocated for sequence " << seq_idx << std::endl;
        return;
    }
    
    std::cout << "Pages allocated for sequence " << seq_idx << ": ";
    for (int page : seq_to_page_mapping[seq_idx]) {
        std::cout << page << " ";
    }
    std::cout << std::endl;
}

} // namespace paged_attention
