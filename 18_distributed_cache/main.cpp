#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <mutex>
#include <thread>

#define NUM_NODES 5
#define CACHE_SIZE 1048576 // 1M entries per node
#define TOTAL_ENTRIES 5000000

struct CacheEntry {
    std::string key;
    int value;
    bool occupied;
};

class DistributedCache {
public:
    DistributedCache(int num_nodes) : num_nodes(num_nodes), cache_entries(num_nodes * CACHE_SIZE) {}

    void bulk_set(const std::vector<std::string>& keys, const std::vector<int>& values) {
        int num_entries = keys.size();
        std::vector<std::thread> threads;

        for (int t = 0; t < std::thread::hardware_concurrency(); ++t) {
            threads.emplace_back([&, t]() {
                for (int i = t; i < num_entries; i += std::thread::hardware_concurrency()) {
                    int node_idx = i % num_nodes;
                    int cache_idx = node_idx * CACHE_SIZE + (i / num_nodes) % CACHE_SIZE;

                    std::lock_guard<std::mutex> lock(cache_mutexes[node_idx]);
                    cache_entries[cache_idx].key = keys[i];
                    cache_entries[cache_idx].value = values[i];
                    cache_entries[cache_idx].occupied = true;
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

private:
    int num_nodes;
    std::vector<CacheEntry> cache_entries;
    std::mutex cache_mutexes[NUM_NODES];
};

int main() {
    DistributedCache cache(NUM_NODES);

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

