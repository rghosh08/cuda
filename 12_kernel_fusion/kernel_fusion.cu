// CUDA Kernel Fusion Example with Temperature Monitoring
// This demonstrates fusing multiple operations into a single kernel
// to reduce memory bandwidth and improve performance

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <nvml.h>
#include <thread>
#include <chrono>
#include <atomic>

// Temperature monitoring globals
std::atomic<bool> keep_monitoring(true);
std::atomic<float> current_temp(0.0f);
std::atomic<float> max_temp(0.0f);

// Function to monitor temperature in background
void monitor_temperature(int device_id) {
    nvmlDevice_t device;
    nvmlReturn_t result;
    
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("NVML init failed, temperature monitoring disabled\n");
        return;
    }
    
    result = nvmlDeviceGetHandleByIndex(device_id, &device);
    if (result != NVML_SUCCESS) {
        nvmlShutdown();
        return;
    }
    
    while (keep_monitoring) {
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (result == NVML_SUCCESS) {
            current_temp = temp;
            if (temp > max_temp) max_temp = temp;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    nvmlShutdown();
}

// Non-fused approach: Three separate kernels
__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void multiply_kernel(float *c, float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = c[idx] * d[idx];
    }
}

__global__ void sqrt_kernel(float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = sqrtf(c[idx]);
    }
}

// Fused kernel: Combines all three operations
__global__ void fused_add_mul_sqrt_kernel(float *a, float *b, float *d, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perform all operations in registers without intermediate memory accesses
        float temp = a[idx] + b[idx];  // Addition
        temp = temp * d[idx];           // Multiplication
        result[idx] = sqrtf(temp);      // Square root
    }
}

// More complex fusion example: GEMM + bias + activation
__global__ void fused_gemm_bias_relu(
    float *A, float *B, float *C, float *bias,
    int M, int N, int K, float alpha, float beta)
{
    // Shared memory for tile-based computation
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Tile-based matrix multiplication
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tiles into shared memory
        if (row < M && t * 16 + tx < K)
            As[ty][tx] = A[row * K + t * 16 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t * 16 + ty < K && col < N)
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Apply scaling, bias, and ReLU activation (fused)
    if (row < M && col < N) {
        sum = alpha * sum + beta * C[row * N + col];  // Scaling
        sum += bias[col];                              // Bias addition
        sum = fmaxf(sum, 0.0f);                       // ReLU activation
        C[row * N + col] = sum;
    }
}

// Reduction + normalization fusion example
__global__ void fused_mean_variance_normalize(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Compute sum for mean
    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / n;
    __syncthreads();
    
    // Phase 2: Compute variance
    float var = 0.0f;
    if (idx < n) {
        float diff = input[idx] - mean;
        var = diff * diff;
    }
    sdata[tid] = var;
    __syncthreads();
    
    // Parallel reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float variance = sdata[0] / n;
    float stddev = sqrtf(variance + 1e-8f);  // Add epsilon for numerical stability
    
    // Phase 3: Normalize (fused with computation)
    if (idx < n) {
        output[idx] = (input[idx] - mean) / stddev;
    }
}

// Host function to demonstrate usage with temperature monitoring
void demonstrate_kernel_fusion() {
    const int N = 100 * 1024 * 1024;  // 100M elements for more heat generation
    const int bytes = N * sizeof(float);
    const int num_runs = 50;  // Multiple runs to generate heat
    
    // Start temperature monitoring thread
    std::thread temp_thread(monitor_temperature, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Let it initialize
    
    printf("Starting kernel fusion demo with temperature monitoring...\n");
    printf("Initial GPU temperature: %.0f°C\n\n", current_temp.load());
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_d = (float*)malloc(bytes);
    float *h_result_fused = (float*)malloc(bytes);
    float *h_result_nonfused = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f + (float)rand() / RAND_MAX;
        h_b[i] = 2.0f + (float)rand() / RAND_MAX;
        h_d[i] = 3.0f + (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_d, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_d, bytes);
    cudaMalloc(&d_result, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_nonfused_time = 0;
    float total_fused_time = 0;
    
    printf("Running non-fused kernels %d times...\n", num_runs);
    float start_temp = current_temp.load();
    
    // Non-fused approach - multiple runs
    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        multiply_kernel<<<gridSize, blockSize>>>(d_c, d_d, N);
        sqrt_kernel<<<gridSize, blockSize>>>(d_c, N);
        
        if (run % 10 == 0) {
            cudaDeviceSynchronize();
            printf("  Run %d/50 - Temp: %.0f°C\n", run, current_temp.load());
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float nonFusedTime;
    cudaEventElapsedTime(&nonFusedTime, start, stop);
    total_nonfused_time = nonFusedTime;
    
    float peak_temp_nonfused = max_temp.load();
    printf("Non-fused peak temperature: %.0f°C (Δ%.0f°C)\n", peak_temp_nonfused, peak_temp_nonfused - start_temp);
    printf("Non-fused total time: %.3f ms\n\n", nonFusedTime);
    
    // Cool down period
    printf("Cooling down...\n");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Reset max temp for fused kernel test
    max_temp = current_temp.load();
    start_temp = current_temp.load();
    
    printf("Running fused kernel %d times...\n", num_runs);
    
    // Fused approach - multiple runs
    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        fused_add_mul_sqrt_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_d, d_result, N);
        
        if (run % 10 == 0) {
            cudaDeviceSynchronize();
            printf("  Run %d/50 - Temp: %.0f°C\n", run, current_temp.load());
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fusedTime;
    cudaEventElapsedTime(&fusedTime, start, stop);
    total_fused_time = fusedTime;
    
    float peak_temp_fused = max_temp.load();
    printf("Fused peak temperature: %.0f°C (Δ%.0f°C)\n", peak_temp_fused, peak_temp_fused - start_temp);
    printf("Fused total time: %.3f ms\n\n", fusedTime);
    
    // Copy one result back for verification
    cudaMemcpy(h_result_fused, d_result, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_nonfused, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results match
    float max_diff = 0.0f;
    for (int i = 0; i < 1000; i++) {  // Check first 1000 elements
        float expected = sqrtf((h_a[i] + h_b[i]) * h_d[i]);
        float diff = fabs(h_result_fused[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("=== Performance Summary ===\n");
    printf("Non-fused time: %.3f ms (%.2f ms per run)\n", total_nonfused_time, total_nonfused_time / num_runs);
    printf("Fused time: %.3f ms (%.2f ms per run)\n", total_fused_time, total_fused_time / num_runs);
    printf("Speedup: %.2fx\n", total_nonfused_time / total_fused_time);
    printf("Max error: %e\n\n", max_diff);
    
    printf("=== Temperature Summary ===\n");
    printf("Non-fused peak temp: %.0f°C\n", peak_temp_nonfused);
    printf("Fused peak temp: %.0f°C\n", peak_temp_fused);
    printf("Temperature difference: %.0f°C\n", fabs(peak_temp_nonfused - peak_temp_fused));
    
    // Stop temperature monitoring
    keep_monitoring = false;
    temp_thread.join();
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_d);
    free(h_result_fused);
    free(h_result_nonfused);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Check if NVML is available
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("WARNING: NVML not available. Temperature monitoring disabled.\n");
        printf("To enable temperature monitoring:\n");
        printf("  Linux: Make sure libnvidia-ml.so is installed\n");
        printf("  Windows: Make sure nvml.dll is in PATH\n\n");
        printf("You can still monitor temperature externally using:\n");
        printf("  watch -n 1 nvidia-smi\n\n");
    } else {
        nvmlShutdown();
    }
    
    demonstrate_kernel_fusion();
    return 0;
}

// Compile with:
// nvcc -o kernel_fusion_temp kernel_fusion.cu -lnvidia-ml -std=c++11 -lpthread
// 
// If NVML linking fails, compile without temperature monitoring:
// nvcc -o kernel_fusion kernel_fusion.cu -DNO_TEMP_MONITOR
