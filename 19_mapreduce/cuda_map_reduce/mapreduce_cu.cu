#include "mapreduce_cu.h"

// CUDA Kernels Implementation
__global__ void map_kernel(int* input, KeyValue* mapped, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        mapped[idx].key = input[idx];
        mapped[idx].value = 1;
    }
}

__global__ void reduce_kernel(KeyValue* input, KeyValue* output, int* counts, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int key = input[idx].key;
        atomicAdd(&counts[key], 1);
    }
}

__global__ void compact_results(int* counts, KeyValue* output, int max_keys, int* result_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < max_keys && counts[idx] > 0) {
        int pos = atomicAdd(result_count, 1);
        output[pos].key = idx;
        output[pos].value = counts[idx];
    }
}

// MapReduce Implementations
void custom_mapreduce(int* data, int n, KeyValue* results, int* num_results) {
    KeyValue *d_mapped, *d_results;
    int *d_data, *d_counts, *d_result_count;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_mapped, n * sizeof(KeyValue)));
    CUDA_CHECK(cudaMalloc(&d_results, MAX_WORDS * sizeof(KeyValue)));
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, MAX_WORDS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counts, 0, MAX_WORDS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(int)));
    
    // Map phase
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    map_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, d_mapped, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Reduce phase
    reduce_kernel<<<grid_size, BLOCK_SIZE>>>(d_mapped, d_results, d_counts, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compact results
    int count_grid = (MAX_WORDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compact_results<<<count_grid, BLOCK_SIZE>>>(d_counts, d_results, MAX_WORDS, d_result_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(num_results, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results, d_results, (*num_results) * sizeof(KeyValue), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_mapped);
    cudaFree(d_results);
    cudaFree(d_data);
    cudaFree(d_counts);
    cudaFree(d_result_count);
}

void thrust_mapreduce(int* data, int n, KeyValue* results, int* num_results) {
    thrust::device_vector<int> d_words(data, data + n);
    thrust::device_vector<int> d_counts(d_words.size(), 1);
    
    // Sort by key for grouping
    thrust::sort_by_key(d_words.begin(), d_words.end(), d_counts.begin());
    
    // Reduce by key
    thrust::device_vector<int> unique_words(n);
    thrust::device_vector<int> word_counts(n);
    
    auto end = thrust::reduce_by_key(d_words.begin(), d_words.end(), d_counts.begin(),
                                     unique_words.begin(), word_counts.begin());
    
    *num_results = end.first - unique_words.begin();
    
    // Copy results
    for (int i = 0; i < *num_results; i++) {
        results[i].key = unique_words[i];
        results[i].value = word_counts[i];
    }
}

// Test Functions
void test_word_count() {
    printf("=== WORD COUNT TEST ===\n");
    
    // Small test dataset
    int words[] = {1, 2, 1, 3, 2, 1, 4, 3, 1, 2};
    int n = sizeof(words) / sizeof(words[0]);
    
    KeyValue results[10];
    int num_results;
    
    // Test custom implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    custom_mapreduce(words, n, results, &num_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    
    printf("Custom Implementation Results:\n");
    for (int i = 0; i < num_results; i++) {
        printf("Word %d: Count %d\n", results[i].key, results[i].value);
    }
    printf("Custom Time: %.3f ms\n\n", custom_time);
    
    // Test Thrust implementation
    cudaEventRecord(start);
    thrust_mapreduce(words, n, results, &num_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, start, stop);
    
    printf("Thrust Implementation Results:\n");
    for (int i = 0; i < num_results; i++) {
        printf("Word %d: Count %d\n", results[i].key, results[i].value);
    }
    printf("Thrust Time: %.3f ms\n\n", thrust_time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test_large_dataset(long long n) {
    printf("=== LARGE DATASET PERFORMANCE TEST ===\n");
    printf("Dataset size: %lld elements\n", n);
    
    // Check memory requirements
    check_memory_requirements(n);
    
    int* data = (int*)malloc(n * sizeof(int));
    if (!data) {
        double memory_gb = (n * sizeof(int)) / (1024.0 * 1024.0 * 1024.0);
        printf("Error: Failed to allocate memory for %lld elements (%.2f GB)\n", n, memory_gb);
        return;
    }
    
    // Generate random data
    srand(time(NULL));
    long long max_unique_keys = (n < 10000) ? n : 10000;  // Limit unique keys to prevent excessive memory usage
    for (long long i = 0; i < n; i++) {
        data[i] = rand() % max_unique_keys;
    }
    
    printf("Unique keys: %lld\n", max_unique_keys);
    
    KeyValue* results = (KeyValue*)malloc(max_unique_keys * sizeof(KeyValue));
    if (!results) {
        printf("Error: Failed to allocate memory for results\n");
        free(data);
        return;
    }
    
    // Check if dataset size exceeds CUDA kernel limitations
    if (n > INT_MAX) {
        printf("Warning: Dataset size (%lld) exceeds CUDA kernel index limits.\n", n);
        printf("Processing first %d elements only.\n", INT_MAX);
        n = INT_MAX;
    }
    
    int dataset_size = (int)n;  // Safe cast after check above
    int num_results;
    
    // Performance comparison
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test custom implementation
    printf("Running custom MapReduce...\n");
    cudaEventRecord(start);
    custom_mapreduce(data, dataset_size, results, &num_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    printf("Custom MapReduce time: %.3f ms\n", custom_time);
    
    // Test Thrust implementation
    printf("Running Thrust MapReduce...\n");
    cudaEventRecord(start);
    thrust_mapreduce(data, dataset_size, results, &num_results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, start, stop);
    printf("Thrust MapReduce time: %.3f ms\n", thrust_time);
    
    if (thrust_time > 0) {
        printf("Speedup: %.2fx\n", custom_time / thrust_time);
        printf("Throughput: %.2f M elements/sec\n", dataset_size / (thrust_time * 1000.0));
    }
    
    free(data);
    free(results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Utility Functions
void print_usage(const char* program_name) {
    printf("Usage: %s [dataset_size]\n", program_name);
    printf("  dataset_size: Number of elements for large dataset test (default: 10000000)\n");
    printf("                Must be between 1000 and 100000000000 (100B)\n");
    printf("                Note: Large datasets require significant GPU memory\n");
    printf("\nExamples:\n");
    printf("  %s                # Run with default 10M elements\n", program_name);
    printf("  %s 1000000        # Run with 1M elements\n", program_name);
    printf("  %s 50000000       # Run with 50M elements\n", program_name);
    printf("  %s 1000000000     # Run with 1B elements (~4GB memory)\n", program_name);
    printf("  %s 10000000000    # Run with 10B elements (~40GB memory)\n", program_name);
}

void print_device_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        exit(1);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Grid Size: (%d, %d, %d)\n\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

void check_memory_requirements(long long n) {
    double memory_gb = (n * sizeof(int)) / (1024.0 * 1024.0 * 1024.0);
    printf("Memory required: %.2f GB\n", memory_gb);
    
    // Get available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
    
    printf("GPU Memory: %.2f GB free / %.2f GB total\n", free_gb, total_gb);
    
    if (memory_gb > free_gb * 0.8) {  // Use 80% threshold for safety
        printf("⚠️  Warning: Dataset requires %.2f GB but only %.2f GB available!\n", 
               memory_gb, free_gb);
        printf("   Consider using a smaller dataset or freeing GPU memory.\n");
    } else if (memory_gb > total_gb * 0.5) {
        printf("⚡ Info: Large dataset will use %.1f%% of GPU memory.\n", 
               (memory_gb / total_gb) * 100.0);
    }
    printf("\n");
}

// Main Function
int main(int argc, char** argv) {
    printf("CUDA MapReduce Implementation\n");
    printf("=============================\n\n");
    
    // Print device information
    print_device_info();
    
    // Parse command line arguments
    long long dataset_size = 10000000LL;  // Default 10M elements
    
    if (argc == 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        
        char* endptr;
        long long parsed_size = strtoll(argv[1], &endptr, 10);
        
        if (*endptr != '\0' || parsed_size < 1000 || parsed_size > 100000000000LL) {
            printf("Error: Invalid dataset size '%s'\n", argv[1]);
            printf("Dataset size must be between 1,000 and 100,000,000,000 (100B)\n\n");
            print_usage(argv[0]);
            return 1;
        }
        
        dataset_size = parsed_size;
    } else if (argc > 2) {
        printf("Error: Too many arguments\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Run tests
    test_word_count();
    test_large_dataset(dataset_size);
    
    printf("MapReduce tests completed successfully!\n");
    return 0;
}
