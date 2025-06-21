#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Key-Value pair structure
struct KeyValue {
    int key;
    int value;
    
    __host__ __device__ KeyValue() : key(0), value(0) {}
    __host__ __device__ KeyValue(int k, int v) : key(k), value(v) {}
};

// Comparison operator for sorting
struct KeyValueComparator {
    __host__ __device__ bool operator()(const KeyValue& a, const KeyValue& b) {
        return a.key < b.key;
    }
};

// Equality operator for grouping
struct KeyValueEqual {
    __host__ __device__ bool operator()(const KeyValue& a, const KeyValue& b) {
        return a.key == b.key;
    }
};

// Addition operator for reduction
struct KeyValueAdd {
    __host__ __device__ KeyValue operator()(const KeyValue& a, const KeyValue& b) {
        return KeyValue(a.key, a.value + b.value);
    }
};

// =============================================================================
// MAP PHASE: Convert input data to key-value pairs
// =============================================================================

__global__ void map_word_count(int* words, int n, KeyValue* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Map each word to (word_id, 1)
        output[idx] = KeyValue(words[idx], 1);
    }
}

__global__ void map_square_sum(int* numbers, int n, KeyValue* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Map each number to (number, number^2)
        int num = numbers[idx];
        output[idx] = KeyValue(num, num * num);
    }
}

// =============================================================================
// REDUCE PHASE: Efficient parallel reduction using shared memory
// =============================================================================

__global__ void reduce_by_key_shared(KeyValue* input, KeyValue* output, 
                                   int* group_starts, int* group_sizes, 
                                   int num_groups) {
    extern __shared__ KeyValue sdata[];
    
    int group_id = blockIdx.x;
    if (group_id >= num_groups) return;
    
    int tid = threadIdx.x;
    int group_start = group_starts[group_id];
    int group_size = group_sizes[group_id];
    
    // Load data into shared memory
    KeyValue sum = KeyValue(0, 0);
    for (int i = tid; i < group_size; i += blockDim.x) {
        if (i < group_size) {
            KeyValue item = input[group_start + i];
            if (sum.key == 0) sum.key = item.key;  // Set key once
            sum.value += item.value;
        }
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            sdata[tid].value += sdata[tid + s].value;
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}

// =============================================================================
// SHUFFLE PHASE: Group by key preparation
// =============================================================================

__global__ void find_group_boundaries(KeyValue* sorted_data, int n, 
                                     int* group_starts, int* group_sizes,
                                     int* num_groups) {
    int idx = threadIdx.x;
    
    if (idx == 0) {
        *num_groups = 0;
        if (n > 0) {
            group_starts[0] = 0;
            int current_group = 0;
            
            for (int i = 1; i < n; i++) {
                if (sorted_data[i].key != sorted_data[i-1].key) {
                    group_sizes[current_group] = i - group_starts[current_group];
                    current_group++;
                    group_starts[current_group] = i;
                }
            }
            group_sizes[current_group] = n - group_starts[current_group];
            *num_groups = current_group + 1;
        }
    }
}

// =============================================================================
// MAPREDUCE FRAMEWORK CLASS
// =============================================================================

class CudaMapReduce {
private:
    KeyValue* d_mapped_data;
    KeyValue* d_sorted_data;
    KeyValue* d_reduced_data;
    int* d_group_starts;
    int* d_group_sizes;
    int* d_num_groups;
    
    int max_size;
    
public:
    CudaMapReduce(int max_elements) : max_size(max_elements) {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_mapped_data, max_size * sizeof(KeyValue)));
        CUDA_CHECK(cudaMalloc(&d_sorted_data, max_size * sizeof(KeyValue)));
        CUDA_CHECK(cudaMalloc(&d_reduced_data, max_size * sizeof(KeyValue)));
        CUDA_CHECK(cudaMalloc(&d_group_starts, max_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_group_sizes, max_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_groups, sizeof(int)));
    }
    
    ~CudaMapReduce() {
        cudaFree(d_mapped_data);
        cudaFree(d_sorted_data);
        cudaFree(d_reduced_data);
        cudaFree(d_group_starts);
        cudaFree(d_group_sizes);
        cudaFree(d_num_groups);
    }
    
    // Generic MapReduce execution
    int execute_word_count(int* input_data, int n, KeyValue* result) {
        // MAP PHASE
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        map_word_count<<<blocksPerGrid, threadsPerBlock>>>(input_data, n, d_mapped_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return reduce_phase(n, result);
    }
    
    int execute_square_sum(int* input_data, int n, KeyValue* result) {
        // MAP PHASE
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        map_square_sum<<<blocksPerGrid, threadsPerBlock>>>(input_data, n, d_mapped_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        return reduce_phase(n, result);
    }
    
private:
    int reduce_phase(int n, KeyValue* result) {
        // SHUFFLE PHASE: Sort by key
        thrust::device_ptr<KeyValue> thrust_mapped(d_mapped_data);
        thrust::device_ptr<KeyValue> thrust_sorted(d_sorted_data);
        
        thrust::copy(thrust_mapped, thrust_mapped + n, thrust_sorted);
        thrust::sort(thrust_sorted, thrust_sorted + n, KeyValueComparator());
        
        // Find group boundaries
        find_group_boundaries<<<1, 1>>>(d_sorted_data, n, d_group_starts, 
                                       d_group_sizes, d_num_groups);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get number of unique keys
        int num_groups;
        CUDA_CHECK(cudaMemcpy(&num_groups, d_num_groups, sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        // REDUCE PHASE: Reduce by key
        int threadsPerBlock = 256;
        size_t sharedMemSize = threadsPerBlock * sizeof(KeyValue);
        
        reduce_by_key_shared<<<num_groups, threadsPerBlock, sharedMemSize>>>(
            d_sorted_data, d_reduced_data, d_group_starts, d_group_sizes, num_groups);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(result, d_reduced_data, num_groups * sizeof(KeyValue),
                             cudaMemcpyDeviceToHost));
        
        return num_groups;
    }
};

// Functor structures for Thrust operations (replaces lambdas)
struct word_to_kv_functor {
    __host__ __device__ KeyValue operator()(int word) const {
        return KeyValue(word, 1);
    }
};

struct square_to_kv_functor {
    __host__ __device__ KeyValue operator()(int num) const {
        return KeyValue(num, num * num);
    }
};

struct extract_key {
    __host__ __device__ int operator()(const KeyValue& kv) const {
        return kv.key;
    }
};

struct extract_value {
    __host__ __device__ int operator()(const KeyValue& kv) const {
        return kv.value;
    }
};

// =============================================================================
// ALTERNATIVE: Using Thrust Library (More Efficient)  
// =============================================================================

class ThrustMapReduce {
public:
    static int word_count_thrust(int* input_data, int n, KeyValue* result) {
        // Create device vectors
        thrust::device_vector<int> d_input(input_data, input_data + n);
        thrust::device_vector<KeyValue> mapped_data(n);
        
        // MAP PHASE: Transform input to key-value pairs
        thrust::transform(d_input.begin(), d_input.end(), mapped_data.begin(), 
                         word_to_kv_functor());
        
        // SHUFFLE PHASE: Sort by key
        thrust::sort(mapped_data.begin(), mapped_data.end(), KeyValueComparator());
        
        // REDUCE PHASE: Reduce by key using different approach
        thrust::device_vector<int> keys(n);
        thrust::device_vector<int> values(n);
        
        // Extract keys and values
        thrust::transform(mapped_data.begin(), mapped_data.end(), keys.begin(), extract_key());
        thrust::transform(mapped_data.begin(), mapped_data.end(), values.begin(), extract_value());
        
        // Reduce by key
        thrust::device_vector<int> unique_keys(n);
        thrust::device_vector<int> reduced_values(n);
        
        auto end = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(),
                                        unique_keys.begin(), reduced_values.begin());
        
        int num_results = end.first - unique_keys.begin();
        
        // Combine results back to KeyValue pairs
        for (int i = 0; i < num_results; i++) {
            result[i] = KeyValue(unique_keys[i], reduced_values[i]);
        }
        
        return num_results;
    }
    
    static int square_sum_thrust(int* input_data, int n, KeyValue* result) {
        thrust::device_vector<int> d_input(input_data, input_data + n);
        thrust::device_vector<KeyValue> mapped_data(n);
        
        // MAP: number -> (number, number^2)
        thrust::transform(d_input.begin(), d_input.end(), mapped_data.begin(),
                         square_to_kv_functor());
        
        thrust::sort(mapped_data.begin(), mapped_data.end(), KeyValueComparator());
        
        // Extract keys and values for reduction
        thrust::device_vector<int> keys(n);
        thrust::device_vector<int> values(n);
        
        thrust::transform(mapped_data.begin(), mapped_data.end(), keys.begin(), extract_key());
        thrust::transform(mapped_data.begin(), mapped_data.end(), values.begin(), extract_value());
        
        thrust::device_vector<int> unique_keys(n);
        thrust::device_vector<int> reduced_values(n);
        
        auto end = thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(),
                                        unique_keys.begin(), reduced_values.begin());
        
        int num_results = end.first - unique_keys.begin();
        
        for (int i = 0; i < num_results; i++) {
            result[i] = KeyValue(unique_keys[i], reduced_values[i]);
        }
        
        return num_results;
    }
};

// =============================================================================
// PERFORMANCE TESTING AND EXAMPLES
// =============================================================================

void test_word_count() {
    printf("=== WORD COUNT TEST ===\n");
    
    // Sample data: word IDs
    int n = 10;
    int words[] = {1, 2, 1, 3, 2, 1, 4, 2, 3, 1};
    
    int *d_words;
    CUDA_CHECK(cudaMalloc(&d_words, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_words, words, n * sizeof(int), cudaMemcpyHostToDevice));
    
    KeyValue* results = (KeyValue*)malloc(n * sizeof(KeyValue));
    
    // Test custom implementation
    CudaMapReduce mapreduce(1000);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    int num_results = mapreduce.execute_word_count(d_words, n, results);
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
    num_results = ThrustMapReduce::word_count_thrust(d_words, n, results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, start, stop);
    
    printf("Thrust Implementation Results:\n");
    for (int i = 0; i < num_results; i++) {
        printf("Word %d: Count %d\n", results[i].key, results[i].value);
    }
    printf("Thrust Time: %.3f ms\n\n", thrust_time);
    
    cudaFree(d_words);
    free(results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test_large_dataset() {
    printf("=== LARGE DATASET PERFORMANCE TEST ===\n");
    
    int n = 10000000;  // 10M elements
    int* data = (int*)malloc(n * sizeof(int));
    
    // Generate random data
    for (int i = 0; i < n; i++) {
        data[i] = rand() % 1000;  // 1000 unique words
    }
    
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice));
    
    KeyValue* results = (KeyValue*)malloc(1000 * sizeof(KeyValue));
    
    // Performance comparison
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Custom implementation
    CudaMapReduce mapreduce(n);
    cudaEventRecord(start);
    int num_results = mapreduce.execute_word_count(d_data, n, results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    
    // Thrust implementation
    cudaEventRecord(start);
    num_results = ThrustMapReduce::word_count_thrust(d_data, n, results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, start, stop);
    
    printf("Dataset size: %d elements\n", n);
    printf("Unique keys: %d\n", num_results);
    printf("Custom MapReduce time: %.3f ms\n", custom_time);
    printf("Thrust MapReduce time: %.3f ms\n", thrust_time);
    printf("Speedup: %.2fx\n", custom_time / thrust_time);
    printf("Throughput: %.2f M elements/sec\n", n / (thrust_time / 1000.0) / 1e6);
    
    free(data);
    free(results);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    printf("CUDA MapReduce Implementation\n");
    printf("=============================\n\n");
    
    // Initialize CUDA
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Run tests
    test_word_count();
    test_large_dataset();
    
    printf("MapReduce tests completed successfully!\n");
    return 0;
}
