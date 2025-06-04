#include <iostream>
#include <vector>
#include <climits>
#include <chrono>
#include <cstdlib>

#define N 1024

int main() {
    std::vector<int> graph(N * N);
    std::vector<bool> inMST(N, false);

    // Random initialization for the adjacency matrix
    for (int i = 0; i < N * N; i++) {
        graph[i] = rand() % 100 + 1;
    }

    inMST[0] = true;

    auto start = std::chrono::high_resolution_clock::now();

    int mst_weight = 0;

    for (int count = 1; count < N; count++) {
        int min_weight = INT_MAX;
        int min_node = -1;

        for (int j = 0; j < N; j++) {
            if (!inMST[j]) {
                for (int i = 0; i < N; i++) {
                    if (inMST[i] && graph[i * N + j] < min_weight) {
                        min_weight = graph[i * N + j];
                        min_node = j;
                    }
                }
            }
        }

        mst_weight += min_weight;
        inMST[min_node] = true;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << "Minimum Spanning Tree Weight: " << mst_weight << "\n";
    std::cout << "Computation Time: " << duration.count() << " ms\n";


    return 0;
}
