#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <functional>

constexpr size_t DATA_SIZE = 1000000000;

void storage_controller_task(const std::vector<int>& data, std::vector<int>& output, std::vector<int>& checksums, size_t start, size_t end, int thread_id) {
    int checksum = 0;
    for (size_t i = start; i < end; ++i) {
        output[i] = data[i] + 1;
        checksum ^= output[i];
    }
    checksums[thread_id] = checksum;
}

int main() {
    const int NUM_THREADS = std::thread::hardware_concurrency();

    std::vector<int> data(DATA_SIZE);
    std::vector<int> output(DATA_SIZE);
    std::vector<int> checksums(NUM_THREADS, 0);

    std::iota(data.begin(), data.end(), 0);

    size_t chunk_size = DATA_SIZE / NUM_THREADS;
    std::vector<std::thread> threads;

    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == NUM_THREADS - 1) ? DATA_SIZE : start + chunk_size;
        threads.emplace_back(storage_controller_task, std::cref(data), std::ref(output), std::ref(checksums), start, end, i);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int final_checksum = std::accumulate(checksums.begin(), checksums.end(), 0, std::bit_xor<int>());

    for (int i = 0; i < 10; ++i) {
        std::cout << "Data[" << i << "]: " << data[i] << " -> " << output[i] << '\n';
    }

    std::cout << "Final checksum: " << final_checksum << std::endl;

    return 0;
}

