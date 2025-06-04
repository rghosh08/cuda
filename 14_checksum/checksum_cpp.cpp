#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <chrono>
#include <cstdlib>

// Thread function to compute partial checksum
void partialChecksum(const std::vector<unsigned char>& data, size_t start, size_t end, unsigned int &partial_sum) {
    partial_sum = std::accumulate(data.begin() + start, data.begin() + end, 0u);
}

int main() {
    const size_t dataSize = 1 << 24; // 16 MB
    std::vector<unsigned char> data(dataSize);

    // Initialize data (simulate read from storage)
    for (size_t i = 0; i < dataSize; ++i)
        data[i] = rand() % 256;

    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    std::vector<unsigned int> partial_sums(num_threads, 0);

    size_t chunk_size = dataSize / num_threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? dataSize : start + chunk_size;
        threads[i] = std::thread(partialChecksum, std::ref(data), start, end, std::ref(partial_sums[i]));
    }

    // Wait for threads to complete
    for (auto& thread : threads)
        thread.join();

    // Compute total checksum
    unsigned int checksum = std::accumulate(partial_sums.begin(), partial_sums.end(), 0u);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    std::cout << "Checksum: " << checksum << "\n";
    std::cout << "Processing Time: " << elapsed.count() << " ms\n";

    return 0;
}

