#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>  // Include OpenMP

const int N = 8192;

// Multithreaded matrix multiplication
void matmul(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int width) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float val = 0.0f;
            for (int k = 0; k < width; ++k) {
                val += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = val;
        }
    }
}

int main() {
    size_t size = N * N;

    std::vector<float> h_A(size, 1.0f);
    std::vector<float> h_B(size, 2.0f);
    std::vector<float> h_C(size, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    matmul(h_A, h_B, h_C, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double total_ops = 2.0 * N * N * N;
    double flops = total_ops / duration.count();

    std::cout << "Matrix multiplication time: " << duration.count() << " seconds\n";
    std::cout << "Performance: " << flops / 1e9 << " GFLOPS\n";
    std::cout << "Sample output C[0]: " << h_C[0] << "\n";

    return 0;
}

