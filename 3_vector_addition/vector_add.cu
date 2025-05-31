#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float *a, float *b, float *c, int n) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < n) {
    	    c[id] = a[id] + b[id];
    	}
}

int main() {
	int n = 1000;
	size_t bytes = n * sizeof(float);

	float *h_a = (float*)malloc(bytes);
	float *h_b = (float*)malloc(bytes);
	float *h_c = (float*)malloc(bytes);

	for (int i = 0; i < n; i++) {
		h_a[i] = 1.0f;
		h_b[i] = 2.0f;
	}


	// Allocate memory on device
    	float *d_a, *d_b, *d_c;
    	cudaMalloc(&d_a, bytes);
    	cudaMalloc(&d_b, bytes);
    	cudaMalloc(&d_c, bytes);

        // Copy data to device
    	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Execute kernel
    	int threads = 256;
    	int blocks = (n + threads - 1) / threads;
    	vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

        // Copy results back to host
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check results
 	for (int i = 0; i < 10; i++)
        	printf("c[%d] = %f\n", i, h_c[i]);

    	// Cleanup
    	cudaFree(d_a);
    	cudaFree(d_b);
    	cudaFree(d_c);
    	free(h_a);
    	free(h_b);
    	free(h_c);

    	return 0;
}

