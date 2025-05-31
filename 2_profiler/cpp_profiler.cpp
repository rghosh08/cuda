#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    size_t size = 1e7;
    std::vector<float> a(size), b(size), c(size);
    
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for(size_t i=0; i<size; i++){
        a[i] = dis(gen);
        b[i] = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<size; i++)
        c[i] = a[i] + b[i];

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;

    std::cout << "Time taken: " << diff.count() << " seconds\n";
    return 0;
}

