#include <iostream>
#include <vector>
#include <chrono>

const int N = 1024;

// Matrix multiplication function
void matmul(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int width) {
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
    size_t bytes = N * N;

    // Allocate host memory
    std::vector<float> h_A(bytes, 1.0f);
    std::vector<float> h_B(bytes, 2.0f);
    std::vector<float> h_C(bytes, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matmul(h_A, h_B, h_C, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Matrix multiplication completed in: " << duration.count() << " milliseconds\n";
    std::cout << "Sample output C[0]: " << h_C[0] << std::endl;

    return 0;
}
