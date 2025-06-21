#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <mutex>
#include <future>
#include <functional>
#include <chrono>
#include <numeric>
#include <atomic>
#include <memory>

// Key-Value pair structure
template<typename K, typename V>
struct KeyValue {
    K key;
    V value;
    
    KeyValue() = default;
    KeyValue(const K& k, const V& v) : key(k), value(v) {}
    
    bool operator<(const KeyValue& other) const {
        return key < other.key;
    }
};

// =============================================================================
// THREAD-SAFE UTILITIES
// =============================================================================

template<typename T>
class ThreadSafeVector {
private:
    std::vector<T> data;
    mutable std::mutex mtx;
    
public:
    void push_back(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        data.push_back(item);
    }
    
    void push_back_batch(const std::vector<T>& items) {
        std::lock_guard<std::mutex> lock(mtx);
        data.insert(data.end(), items.begin(), items.end());
    }
    
    std::vector<T> get_copy() const {
        std::lock_guard<std::mutex> lock(mtx);
        return data;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return data.size();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        data.clear();
    }
};

// =============================================================================
// MAP-REDUCE FRAMEWORK
// =============================================================================

template<typename InputType, typename KeyType, typename ValueType>
class CppMapReduce {
public:
    using KV = KeyValue<KeyType, ValueType>;
    using MapFunction = std::function<std::vector<KV>(const InputType&)>;
    using ReduceFunction = std::function<ValueType(const std::vector<ValueType>&)>;
    
private:
    size_t num_threads;
    ThreadSafeVector<KV> intermediate_results;
    
public:
    CppMapReduce(size_t threads = std::thread::hardware_concurrency()) 
        : num_threads(threads) {}
    
    // Execute MapReduce with custom map and reduce functions
    std::vector<KV> execute(const std::vector<InputType>& input,
                           MapFunction map_func,
                           ReduceFunction reduce_func) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // MAP PHASE
        std::cout << "Starting MAP phase with " << num_threads << " threads...\n";
        map_phase(input, map_func);
        
        auto map_time = std::chrono::high_resolution_clock::now();
        
        // SHUFFLE PHASE
        std::cout << "Starting SHUFFLE phase...\n";
        auto grouped_data = shuffle_phase();
        
        auto shuffle_time = std::chrono::high_resolution_clock::now();
        
        // REDUCE PHASE
        std::cout << "Starting REDUCE phase...\n";
        auto results = reduce_phase(grouped_data, reduce_func);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Print timing information
        auto map_duration = std::chrono::duration_cast<std::chrono::milliseconds>(map_time - start_time);
        auto shuffle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shuffle_time - map_time);
        auto reduce_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - shuffle_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "MAP phase: " << map_duration.count() << " ms\n";
        std::cout << "SHUFFLE phase: " << shuffle_duration.count() << " ms\n";
        std::cout << "REDUCE phase: " << reduce_duration.count() << " ms\n";
        std::cout << "Total time: " << total_duration.count() << " ms\n";
        
        return results;
    }
    
private:
    // MAP PHASE: Parallel processing of input data
    void map_phase(const std::vector<InputType>& input, MapFunction map_func) {
        intermediate_results.clear();
        
        size_t chunk_size = (input.size() + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, input.size());
            
            if (start < input.size()) {
                auto future = std::async(std::launch::async, [this, &input, map_func, start, end]() {
                    std::vector<KV> local_results;
                    
                    for (size_t j = start; j < end; ++j) {
                        auto mapped = map_func(input[j]);
                        local_results.insert(local_results.end(), mapped.begin(), mapped.end());
                    }
                    
                    intermediate_results.push_back_batch(local_results);
                });
                
                futures.push_back(std::move(future));
            }
        }
        
        // Wait for all mapping tasks to complete
        for (auto& future : futures) {
            future.get();
        }
    }
    
    // SHUFFLE PHASE: Group by key
    std::unordered_map<KeyType, std::vector<ValueType>> shuffle_phase() {
        auto all_results = intermediate_results.get_copy();
        std::unordered_map<KeyType, std::vector<ValueType>> grouped;
        
        for (const auto& kv : all_results) {
            grouped[kv.key].push_back(kv.value);
        }
        
        return grouped;
    }
    
    // REDUCE PHASE: Parallel reduction by key
    std::vector<KV> reduce_phase(const std::unordered_map<KeyType, std::vector<ValueType>>& grouped,
                                ReduceFunction reduce_func) {
        
        std::vector<std::pair<KeyType, std::vector<ValueType>>> key_value_pairs(grouped.begin(), grouped.end());
        
        size_t chunk_size = (key_value_pairs.size() + num_threads - 1) / num_threads;
        std::vector<std::future<std::vector<KV>>> futures;
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, key_value_pairs.size());
            
            if (start < key_value_pairs.size()) {
                auto future = std::async(std::launch::async, [&key_value_pairs, reduce_func, start, end]() {
                    std::vector<KV> local_results;
                    
                    for (size_t j = start; j < end; ++j) {
                        const auto& key = key_value_pairs[j].first;
                        const auto& values = key_value_pairs[j].second;
                        ValueType reduced_value = reduce_func(values);
                        local_results.emplace_back(key, reduced_value);
                    }
                    
                    return local_results;
                });
                
                futures.push_back(std::move(future));
            }
        }
        
        // Collect all results
        std::vector<KV> final_results;
        for (auto& future : futures) {
            auto partial_results = future.get();
            final_results.insert(final_results.end(), partial_results.begin(), partial_results.end());
        }
        
        // Sort results by key for consistent output
        std::sort(final_results.begin(), final_results.end());
        
        return final_results;
    }
};

// =============================================================================
// SPECIALIZED MAPREDUCE IMPLEMENTATIONS
// =============================================================================

class WordCountMapReduce {
public:
    using WC_MapReduce = CppMapReduce<int, int, int>;
    
    static std::vector<KeyValue<int, int>> execute(const std::vector<int>& words, size_t num_threads = 0) {
        if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
        
        WC_MapReduce mapreduce(num_threads);
        
        // Map function: word -> (word, 1)
        auto map_func = [](const int& word) -> std::vector<KeyValue<int, int>> {
            return {KeyValue<int, int>(word, 1)};
        };
        
        // Reduce function: sum all counts for each word
        auto reduce_func = [](const std::vector<int>& values) -> int {
            return std::accumulate(values.begin(), values.end(), 0);
        };
        
        return mapreduce.execute(words, map_func, reduce_func);
    }
};

class SquareSumMapReduce {
public:
    using SS_MapReduce = CppMapReduce<int, int, long long>;
    
    static std::vector<KeyValue<int, long long>> execute(const std::vector<int>& numbers, size_t num_threads = 0) {
        if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
        
        SS_MapReduce mapreduce(num_threads);
        
        // Map function: number -> (number, number^2)
        auto map_func = [](const int& num) -> std::vector<KeyValue<int, long long>> {
            return {KeyValue<int, long long>(num, static_cast<long long>(num) * num)};
        };
        
        // Reduce function: sum all squares for each number
        auto reduce_func = [](const std::vector<long long>& values) -> long long {
            return std::accumulate(values.begin(), values.end(), 0LL);
        };
        
        return mapreduce.execute(numbers, map_func, reduce_func);
    }
};

// =============================================================================
// PERFORMANCE OPTIMIZED VERSION (Lock-Free)
// =============================================================================

template<typename InputType, typename KeyType, typename ValueType>
class OptimizedMapReduce {
public:
    using KV = KeyValue<KeyType, ValueType>;
    using MapFunction = std::function<std::vector<KV>(const InputType&)>;
    using ReduceFunction = std::function<ValueType(const std::vector<ValueType>&)>;
    
private:
    size_t num_threads;
    
public:
    OptimizedMapReduce(size_t threads = std::thread::hardware_concurrency()) 
        : num_threads(threads) {}
    
    std::vector<KV> execute(const std::vector<InputType>& input,
                           MapFunction map_func,
                           ReduceFunction reduce_func) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // MAP PHASE - Each thread works on its own chunk
        std::vector<std::vector<KV>> thread_results(num_threads);
        std::vector<std::thread> threads;
        
        size_t chunk_size = (input.size() + num_threads - 1) / num_threads;
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, input.size());
            
            if (start < input.size()) {
                threads.emplace_back([&, i, start, end]() {
                    for (size_t j = start; j < end; ++j) {
                        auto mapped = map_func(input[j]);
                        thread_results[i].insert(thread_results[i].end(), mapped.begin(), mapped.end());
                    }
                });
            }
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto map_time = std::chrono::high_resolution_clock::now();
        
        // SHUFFLE PHASE - Combine and group results
        std::unordered_map<KeyType, std::vector<ValueType>> grouped;
        for (const auto& thread_result : thread_results) {
            for (const auto& kv : thread_result) {
                grouped[kv.key].push_back(kv.value);
            }
        }
        
        auto shuffle_time = std::chrono::high_resolution_clock::now();
        
        // REDUCE PHASE - Parallel reduction
        std::vector<std::pair<KeyType, std::vector<ValueType>>> pairs(grouped.begin(), grouped.end());
        std::vector<KV> results(pairs.size());
        
        threads.clear();
        size_t pairs_per_thread = (pairs.size() + num_threads - 1) / num_threads;
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * pairs_per_thread;
            size_t end = std::min(start + pairs_per_thread, pairs.size());
            
            if (start < pairs.size()) {
                threads.emplace_back([&, start, end, reduce_func]() {
                    for (size_t j = start; j < end; ++j) {
                        ValueType reduced_value = reduce_func(pairs[j].second);
                        results[j] = KV(pairs[j].first, reduced_value);
                    }
                });
            }
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Sort results
        std::sort(results.begin(), results.end());
        
        // Print timing
        auto map_duration = std::chrono::duration_cast<std::chrono::milliseconds>(map_time - start_time);
        auto shuffle_duration = std::chrono::duration_cast<std::chrono::milliseconds>(shuffle_time - map_time);
        auto reduce_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - shuffle_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Optimized - MAP: " << map_duration.count() << " ms, ";
        std::cout << "SHUFFLE: " << shuffle_duration.count() << " ms, ";
        std::cout << "REDUCE: " << reduce_duration.count() << " ms, ";
        std::cout << "Total: " << total_duration.count() << " ms\n";
        
        return results;
    }
};

// =============================================================================
// TESTING AND EXAMPLES
// =============================================================================

void test_word_count() {
    std::cout << "=== WORD COUNT TEST ===\n";
    
    std::vector<int> words = {1, 2, 1, 3, 2, 1, 4, 2, 3, 1, 5, 2, 1, 3, 4};
    
    // Test standard implementation
    auto results = WordCountMapReduce::execute(words);
    
    std::cout << "Word Count Results:\n";
    for (const auto& kv : results) {
        std::cout << "Word " << kv.key << ": " << kv.value << " times\n";
    }
    std::cout << "\n";
    
    // Test optimized implementation
    std::cout << "Optimized version:\n";
    OptimizedMapReduce<int, int, int> opt_mapreduce;
    
    auto map_func = [](const int& word) -> std::vector<KeyValue<int, int>> {
        return {KeyValue<int, int>(word, 1)};
    };
    
    auto reduce_func = [](const std::vector<int>& values) -> int {
        return std::accumulate(values.begin(), values.end(), 0);
    };
    
    auto opt_results = opt_mapreduce.execute(words, map_func, reduce_func);
    
    std::cout << "Optimized Word Count Results:\n";
    for (const auto& kv : opt_results) {
        std::cout << "Word " << kv.key << ": " << kv.value << " times\n";
    }
    std::cout << "\n";
}

void test_large_dataset() {
    std::cout << "=== LARGE DATASET PERFORMANCE TEST ===\n";
    
    const size_t dataset_size = 10000000;
    const int num_unique_words = 1000;
    
    std::vector<int> large_dataset(dataset_size);
    
    // Generate random data
    std::srand(42);  // Fixed seed for reproducibility
    for (size_t i = 0; i < dataset_size; ++i) {
        large_dataset[i] = std::rand() % num_unique_words;
    }
    
    std::cout << "Dataset size: " << dataset_size << " elements\n";
    std::cout << "Unique words: ~" << num_unique_words << "\n";
    std::cout << "Available CPU cores: " << std::thread::hardware_concurrency() << "\n\n";
    
    // Test different thread counts
    std::vector<size_t> thread_counts = {1, 2, 4, 8, std::thread::hardware_concurrency()};
    
    for (size_t threads : thread_counts) {
        if (threads > std::thread::hardware_concurrency()) continue;
        
        std::cout << "Testing with " << threads << " threads:\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = WordCountMapReduce::execute(large_dataset, threads);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double throughput = dataset_size / (duration.count() / 1000.0) / 1e6;
        
        std::cout << "Time: " << duration.count() << " ms\n";
        std::cout << "Throughput: " << throughput << " M elements/sec\n";
        std::cout << "Unique keys found: " << results.size() << "\n\n";
    }
}

void test_square_sum() {
    std::cout << "=== SQUARE SUM TEST ===\n";
    
    std::vector<int> numbers = {1, 2, 3, 2, 1, 4, 3, 2, 1};
    
    auto results = SquareSumMapReduce::execute(numbers);
    
    std::cout << "Square Sum Results:\n";
    for (const auto& kv : results) {
        std::cout << "Number " << kv.key << ": sum of squares = " << kv.value << "\n";
    }
    std::cout << "\n";
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================

int main() {
    std::cout << "C++ MapReduce Implementation\n";
    std::cout << "============================\n\n";
    
    std::cout << "System Info:\n";
    std::cout << "CPU cores: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "Pointer size: " << sizeof(void*) * 8 << "-bit\n\n";
    
    try {
        test_word_count();
        test_square_sum(); 
        test_large_dataset();
        
        std::cout << "All tests completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
