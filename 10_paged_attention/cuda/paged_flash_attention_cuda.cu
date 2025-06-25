// paged_flash_attention_cuda.cu
#include "paged_flash_attention_cuda.h"
#include <cmath>
#include <algorithm>

namespace paged_attention {

// Simple CUDA kernel for initializing page table
__global__ void init_page_table_kernel(int* page_table, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        page_table[idx] = -1;
    }
}

// Simple CUDA kernel for updating KV cache
__global__ void update_kv_cache_kernel(
    float* key_cache,
    float* value_cache,
    const float* keys,
    const float* values,
    const int* page_table,
    int seq_idx,
    int seq_len,
    int max_seq_len,
    int num_heads,
    int head_dim,
    int page_size
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= seq_len || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    int page_idx = page_table[seq_idx * max_seq_len + token_idx];
    if (page_idx < 0) {
        return;
    }
    
    int offset = token_idx % page_size;
    
    size_t cache_idx = (page_idx * page_size + offset) * num_heads * head_dim + 
                      head_idx * head_dim + dim_idx;
                      
    size_t input_idx = token_idx * num_heads * head_dim + 
                      head_idx * head_dim + dim_idx;
    
    key_cache[cache_idx] = keys[input_idx];
    value_cache[cache_idx] = values[input_idx];
}

// Extremely simple attention kernel - focus on correctness
__global__ void simple_attention_kernel(
    const float* key_cache,
    const float* value_cache,
    const float* queries,
    float* outputs,
    const int* page_table,
    const int* seq_lengths,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len,
    int page_size,
    float scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    if (batch_idx >= batch_size || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    int seq_len = seq_lengths[batch_idx];
    if (seq_len <= 0) {
        return;
    }
    // Use global memory for scores
    float* scores = (float*)malloc(seq_len * sizeof(float));
    if (!scores) {
        if (dim_idx == 0 && head_idx == 0 && batch_idx == 0) {
            printf("[ERROR] Failed to allocate global memory for scores (seq_len=%d)\n", seq_len);
        }
        return;
    }
    int q_idx = (batch_idx * num_heads + head_idx) * head_dim + dim_idx;
    int out_idx = q_idx;
    float output_val = 0.0f;
    float max_score = -INFINITY;
    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        int page_idx = page_table[batch_idx * max_seq_len + token_idx];
        if (page_idx < 0) {
            scores[token_idx] = -INFINITY;
            continue;
        }
        int offset = token_idx % page_size;
        float dot_product = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int k_idx = (page_idx * page_size + offset) * num_heads * head_dim + 
                       head_idx * head_dim + d;
            int q_d_idx = (batch_idx * num_heads + head_idx) * head_dim + d;
            dot_product += queries[q_d_idx] * key_cache[k_idx];
        }
        float score = dot_product * scale;
        scores[token_idx] = score;
        if (score > max_score) {
            max_score = score;
        }
    }
    float sum_exp = 0.0f;
    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        if (scores[token_idx] == -INFINITY) {
            continue;
        }
        int page_idx = page_table[batch_idx * max_seq_len + token_idx];
        int offset = token_idx % page_size;
        float exp_score = expf(scores[token_idx] - max_score);
        sum_exp += exp_score;
        int v_idx = (page_idx * page_size + offset) * num_heads * head_dim + 
                   head_idx * head_dim + dim_idx;
        output_val += exp_score * value_cache[v_idx];
    }
    if (sum_exp > 0.0f) {
        outputs[out_idx] = output_val / sum_exp;
    } else {
        outputs[out_idx] = 0.0f;
    }
    free(scores);
}

PagedFlashAttentionCUDA::PagedFlashAttentionCUDA(const PagedAttentionConfig& config)
    : config(config),
      d_key_cache(nullptr),
      d_value_cache(nullptr),
      d_page_table(nullptr) {
    
    // Initialize with provided configuration
    initialize(config);
}

PagedFlashAttentionCUDA::~PagedFlashAttentionCUDA() {
    free_device_memory();
}

bool PagedFlashAttentionCUDA::initialize(const PagedAttentionConfig& new_config) {
    // Update configuration
    config = new_config;
    
    // Free existing memory if any
    free_device_memory();
    
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
    
    // Allocate device memory
    if (!allocate_device_memory()) {
        std::cerr << "Failed to allocate device memory for PagedFlashAttention" << std::endl;
        return false;
    }
    
    // Print memory usage info
    print_memory_usage();
    
    std::cout << "✅ PagedFlashAttention CUDA initialized successfully!" << std::endl;
    return true;
}

bool PagedFlashAttentionCUDA::allocate_device_memory() {
    try {
        // Calculate memory sizes
        size_t kv_cache_size = static_cast<size_t>(config.num_pages) * 
                              config.page_size * 
                              config.num_heads * 
                              config.head_dim;
        
        size_t page_table_size = static_cast<size_t>(config.batch_size) * 
                                config.max_seq_len;
        
        // Allocate key-value cache
        cudaError_t err = cudaMalloc(&d_key_cache, kv_cache_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate key cache: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_value_cache, kv_cache_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate value cache: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_key_cache);
            d_key_cache = nullptr;
            return false;
        }
        
        // Allocate page table
        err = cudaMalloc(&d_page_table, page_table_size * sizeof(int));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate page table: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_key_cache);
            cudaFree(d_value_cache);
            d_key_cache = nullptr;
            d_value_cache = nullptr;
            return false;
        }
        
        // Initialize page table with -1
        int threads_per_block = 256;
        int num_blocks = (page_table_size + threads_per_block - 1) / threads_per_block;
        
        init_page_table_kernel<<<num_blocks, threads_per_block>>>(
            d_page_table, page_table_size
        );
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Error initializing page table: " << cudaGetErrorString(err) << std::endl;
            free_device_memory();
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during device memory allocation: " << e.what() << std::endl;
        free_device_memory();
        return false;
    }
}

void PagedFlashAttentionCUDA::free_device_memory() {
    if (d_key_cache) cudaFree(d_key_cache);
    if (d_value_cache) cudaFree(d_value_cache);
    if (d_page_table) cudaFree(d_page_table);
    
    d_key_cache = nullptr;
    d_value_cache = nullptr;
    d_page_table = nullptr;
}

void PagedFlashAttentionCUDA::print_memory_usage() const {
    // Calculate memory usage in bytes
    size_t kv_cache_bytes = 2 * static_cast<size_t>(config.num_pages) * 
                           config.page_size * 
                           config.num_heads * 
                           config.head_dim * 
                           sizeof(float);
    
    size_t page_table_bytes = static_cast<size_t>(config.batch_size) * 
                             config.max_seq_len * 
                             sizeof(int);
    
    // Convert to KB/MB for display
    double kv_cache_mb = kv_cache_bytes / (1024.0 * 1024.0);
    double page_table_mb = page_table_bytes / (1024.0 * 1024.0);
    
    std::cout << "GPU Memory allocation:" << std::endl;
    std::cout << "  Key/Value cache: " << kv_cache_mb << " MB" << std::endl;
    std::cout << "  Page table: " << page_table_mb << " MB" << std::endl;
    std::cout << "  Total GPU memory: " << (kv_cache_mb + page_table_mb) << " MB" << std::endl;
}

bool PagedFlashAttentionCUDA::allocate_pages_for_sequence(int seq_idx, int num_tokens) {
    if (seq_idx < 0 || seq_idx >= config.batch_size) {
        std::cerr << "Invalid sequence index: " << seq_idx << std::endl;
        return false;
    }
    
    // Calculate number of pages needed
    int pages_needed = (num_tokens + config.page_size - 1) / config.page_size;
    
    // Check if we have enough free pages
    if (free_page_indices.size() < pages_needed) {
        if (config.recycle_pages) {
            // For simplicity, just free all pages from all sequences
            free_page_indices.clear();
            for (int i = 0; i < config.num_pages; i++) {
                free_page_indices.push_back(i);
            }
            
            for (auto& mapping : seq_to_page_mapping) {
                mapping.clear();
            }
        }
        
        // Check again after recycling
        if (free_page_indices.size() < pages_needed) {
            std::cerr << "Out of pages: needed " << pages_needed 
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

bool PagedFlashAttentionCUDA::update_kv_cache(
    int seq_idx, int seq_len, const float* h_keys, const float* h_values) {
    
    std::cout << "Updating KV cache for sequence " << seq_idx << " (length: " << seq_len << ")" << std::endl;
    
    // Allocate pages for this sequence
    if (!allocate_pages_for_sequence(seq_idx, seq_len)) {
        return false;
    }
    
    // Get allocated pages for this sequence
    const auto& allocated_pages = seq_to_page_mapping[seq_idx];
    
    // Update page table on host
    std::vector<int> h_page_table_section(seq_len, -1);
    
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
        h_page_table_section[token_idx] = physical_page;
        
        // Update counters
        page_offset++;
    }
    
    // Copy updated page table section to device
    cudaError_t err = cudaMemcpy(
        d_page_table + seq_idx * config.max_seq_len,
        h_page_table_section.data(),
        seq_len * sizeof(int),
        cudaMemcpyHostToDevice
    );
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to update page table: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy keys and values to device
    float* d_keys;
    float* d_values;
    
    size_t kv_size = seq_len * config.num_heads * config.head_dim * sizeof(float);
    
    err = cudaMalloc(&d_keys, kv_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for keys: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_values, kv_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for values: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_keys);
        return false;
    }
    
    err = cudaMemcpy(d_keys, h_keys, kv_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy keys to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_keys);
        cudaFree(d_values);
        return false;
    }
    
    err = cudaMemcpy(d_values, h_values, kv_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy values to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_keys);
        cudaFree(d_values);
        return false;
    }
    
    // Update KV cache on device
    dim3 grid(seq_len, config.num_heads);
    dim3 block(std::min(config.head_dim, 32)); // Limit to 32 threads for simplicity
    
    update_kv_cache_kernel<<<grid, block>>>(
        d_key_cache,
        d_value_cache,
        d_keys,
        d_values,
        d_page_table,
        seq_idx,
        seq_len,
        config.max_seq_len,
        config.num_heads,
        config.head_dim,
        config.page_size
    );
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in update_kv_cache_kernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_keys);
        cudaFree(d_values);
        return false;
    }
    
    // Free temporary device memory
    cudaFree(d_keys);
    cudaFree(d_values);
    
    std::cout << "  Updated " << allocated_pages.size() << " pages with " 
              << seq_len << " tokens" << std::endl;
    
    // After updating KV cache for all sequences
    std::cout << "Total pages allocated after batch: ";
    int total_pages = 0;
    for (const auto& mapping : seq_to_page_mapping) {
        total_pages += mapping.size();
    }
    std::cout << total_pages << " / " << config.num_pages << " pages used." << std::endl;
    
    print_memory_usage();
    
    return true;
}

bool PagedFlashAttentionCUDA::compute_attention(
    const float* h_queries, float* h_outputs, const int* h_seq_lengths, float scale) {
    
    std::cout << "Computing attention..." << std::endl;
    
    // Copy queries and sequence lengths to device
    float* d_queries;
    int* d_seq_lengths;
    float* d_outputs;
    
    size_t queries_size = config.batch_size * config.num_heads * config.head_dim * sizeof(float);
    size_t seq_lengths_size = config.batch_size * sizeof(int);
    
    cudaError_t err = cudaMalloc(&d_queries, queries_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for queries: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_seq_lengths, seq_lengths_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for sequence lengths: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        return false;
    }
    
    err = cudaMalloc(&d_outputs, queries_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for outputs: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        return false;
    }
    
    err = cudaMemcpy(d_queries, h_queries, queries_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy queries to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    
    err = cudaMemcpy(d_seq_lengths, h_seq_lengths, seq_lengths_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy sequence lengths to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    
    // Initialize outputs to zero
    err = cudaMemset(d_outputs, 0, queries_size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize outputs: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    
    // Launch kernel
    dim3 grid(config.batch_size, config.num_heads);
    dim3 block(config.head_dim);
    size_t shared_mem_size = config.max_seq_len * sizeof(float);
    std::cout << "  Launching kernel..." << std::endl;
    simple_attention_kernel<<<grid, block, shared_mem_size>>>(
        d_key_cache,
        d_value_cache,
        d_queries,
        d_outputs,
        d_page_table,
        d_seq_lengths,
        config.batch_size,
        config.num_heads,
        config.head_dim,
        config.max_seq_len,
        config.page_size,
        scale
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error launching attention kernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in attention kernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    std::cout << "  Kernel completed successfully" << std::endl;
    
    // Copy results back to host
    err = cudaMemcpy(h_outputs, d_outputs, queries_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy outputs from device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_queries);
        cudaFree(d_seq_lengths);
        cudaFree(d_outputs);
        return false;
    }
    
    // Free temporary device memory
    cudaFree(d_queries);
    cudaFree(d_seq_lengths);
    cudaFree(d_outputs);
    
    std::cout << "  Attention computation completed successfully" << std::endl;
    return true;
}

} // namespace paged_attention
