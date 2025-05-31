#include <stdio.h>

__global__ void hello_from_gpu() {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Hello from thread %d!\n", threadId);
}

int main() {

	hello_from_gpu<<<2, 4>>>();
	cudaDeviceSynchronize();
	return 0;
}
