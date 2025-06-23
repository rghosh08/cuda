#ifndef MAPREDUCE_CU_H
#define MAPREDUCE_CU_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Constants
#define BLOCK_SIZE 256
#define MAX_WORDS 10000

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Data structures
struct KeyValue {
    int key;
    int value;
};

// CUDA kernel declarations
__global__ void map_kernel(int* input, KeyValue* mapped, int n);
__global__ void reduce_kernel(KeyValue* input, KeyValue* output, int* counts, int n);
__global__ void compact_results(int* counts, KeyValue* output, int max_keys, int* result_count);

// MapReduce function declarations
void custom_mapreduce(int* data, int n, KeyValue* results, int* num_results);
void thrust_mapreduce(int* data, int n, KeyValue* results, int* num_results);

// Test function declarations
void test_word_count();
void test_large_dataset(long long n);

// Utility function declarations
void print_usage(const char* program_name);
void print_device_info();
void check_memory_requirements(long long n);

#endif // MAPREDUCE_CU_H
