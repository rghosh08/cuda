#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <random>

#define NUM_NODES 5
#define CACHE_SIZE 1048576 // 1M entries per node
#define TOTAL_ENTRIES 5000000

struct CacheEntry {
    char key[32];
    int value;
    bool occupied;
};

__device__ void device_strncpy(char *dest, const char *src, int n) {
    int i;
    for (i = 0; i < n && src[i] != '\0'; i++)
        dest[i] = src[i];
    for (; i < n; i++)
        dest[i] = '\0';
}

__global__ void set_kernel(CacheEntry *cache, const char *keys, int *values, int num_entries, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        int node_idx = idx % num_nodes;
        int cache_idx = node_idx * CACHE_SIZE + (idx / num_nodes) % CACHE_SIZE;
        device_strncpy(cache[cache_idx].key, &keys[idx * 32], 32);
        cache[cache_idx].value = values[idx];
        cache[cache_idx].occupied = true;
    }
}

class DistributedCacheCUDA {
public:
    DistributedCacheCUDA(int num_nodes) : num_nodes(num_nodes) {
        cudaMalloc(&cache_entries, num_nodes * CACHE_SIZE * sizeof(CacheEntry));
        cudaMemset(cache_entries, 0, num_nodes * CACHE_SIZE * sizeof(CacheEntry));
    }

    ~DistributedCacheCUDA() {
        cudaFree(cache_entries);
    }

    void bulk_set(const std::vector<std::string>& keys, const std::vector<int>& values) {
        int num_entries = keys.size();

        char *d_keys;
        int *d_values;

        cudaMalloc(&d_keys, num_entries * 32 * sizeof(char));
        cudaMalloc(&d_values, num_entries * sizeof(int));

        std::vector<char> flat_keys(num_entries * 32);
        for (int i = 0; i < num_entries; ++i)
            strncpy(&flat_keys[i * 32], keys[i].c_str(), 32);

        cudaMemcpy(d_keys, flat_keys.data(), num_entries * 32 * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, values.data(), num_entries * sizeof(int), cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (num_entries + threads - 1) / threads;
        set_kernel<<<blocks, threads>>>(cache_entries, d_keys, d_values, num_entries, num_nodes);
        cudaDeviceSynchronize();

        cudaFree(d_keys);
        cudaFree(d_values);
    }

private:
    CacheEntry *cache_entries;
    int num_nodes;
};

int main() {
    DistributedCacheCUDA cache(NUM_NODES);

    std::vector<std::string> keys(TOTAL_ENTRIES);
    std::vector<int> values(TOTAL_ENTRIES);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 1000000);

    for (int i = 0; i < TOTAL_ENTRIES; ++i) {
        keys[i] = "key_" + std::to_string(i);
        values[i] = dist(rng);
    }

    cache.bulk_set(keys, values);

    std::cout << "Inserted " << TOTAL_ENTRIES << " random entries into the cache." << std::endl;

    return 0;
}

