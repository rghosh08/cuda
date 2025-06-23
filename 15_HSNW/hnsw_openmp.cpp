#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <cstdlib>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [num_nodes] [dim]\n";
        return 1;
    }

    size_t num_nodes = std::stoull(argv[1]);
    size_t dim = std::stoull(argv[2]);

    std::vector<float> nodes(num_nodes * dim);
    std::vector<float> query(dim);
    std::vector<float> distances(num_nodes);

    // Random initialization
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto &val : nodes) val = dist(rng);
    for (auto &val : query) val = dist(rng);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t idx = 0; idx < num_nodes; ++idx) {
        float distance = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float diff = nodes[idx * dim + d] - query[d];
            distance += diff * diff;
        }
        distances[idx] = std::sqrt(distance);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Print sample distances
    std::cout << "Sample distances computed:\n";
    for (int i = 0; i < std::min<size_t>(5, num_nodes); i++) {
        std::cout << "Node " << i << ": " << distances[i] << '\n';
    }

    std::cout << "Computation Time: " << duration.count() << " ms\n";

    return 0;
}
