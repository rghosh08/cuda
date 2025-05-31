#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <chrono>
#include <atomic>

#define LR 0.5f
#define EPOCHS 1000

// Temperature monitoring globals
std::atomic<bool> keep_monitoring(true);
std::atomic<float> current_temp(0.0f);
std::atomic<float> max_temp(0.0f);
std::atomic<float> min_temp(999.0f);
std::atomic<unsigned int> power_usage(0);

// Function to monitor GPU temperature in background
void monitor_gpu_stats(int device_id) {
    nvmlDevice_t device;
    nvmlReturn_t result;
    
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("NVML init failed. Temperature monitoring disabled.\n");
        printf("To enable: ensure libnvidia-ml.so is installed (Linux) or nvml.dll (Windows)\n");
        return;
    }
    
    result = nvmlDeviceGetHandleByIndex(device_id, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle\n");
        nvmlShutdown();
        return;
    }
    
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    printf("Monitoring GPU: %s\n", name);
    
    while (keep_monitoring) {
        unsigned int temp;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        if (result == NVML_SUCCESS) {
            current_temp = temp;
            if (temp > max_temp) max_temp = temp;
            if (temp < min_temp) min_temp = temp;
        }
        
        unsigned int power;
        result = nvmlDeviceGetPowerUsage(device, &power);
        if (result == NVML_SUCCESS) {
            power_usage = power;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    nvmlShutdown();
}

// Sigmoid activation
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

// Forward pass
__global__ void forward(float* x, float* W1, float* b1, float* W2, float* b2, float* h_out, float* y_pred) {
    int idx = threadIdx.x;
    if (idx < 3) {
        float sum = b1[idx];
        for (int i = 0; i < 2; i++)
            sum += x[i] * W1[i * 3 + idx];
        h_out[idx] = sigmoid(sum);
    }

    __syncthreads();

    if (idx == 0) {
        float sum = b2[0];
        for (int i = 0; i < 3; i++)
            sum += h_out[i] * W2[i];
        y_pred[0] = sigmoid(sum);
    }
}

// Backpropagation
__global__ void backward(float* x, float* y, float* h_out, float* y_pred,
                         float* W1, float* b1, float* W2, float* b2) {
    int idx = threadIdx.x;

    float output = y_pred[0];
    float err_output = (output - y[0]) * sigmoid_deriv(output);

    __shared__ float h_errors[3];

    if (idx < 3)
        h_errors[idx] = W2[idx] * err_output * sigmoid_deriv(h_out[idx]);

    __syncthreads();

    if (idx < 3)
        W2[idx] -= LR * err_output * h_out[idx];

    if (idx == 0)
        b2[0] -= LR * err_output;

    __syncthreads();

    if (idx < 3) {
        for (int i = 0; i < 2; i++)
            W1[i * 3 + idx] -= LR * h_errors[idx] * x[i];
        b1[idx] -= LR * h_errors[idx];
    }
}

// Intensive computation kernel to generate more heat
__global__ void intensive_computation(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = data[idx];
        for (int i = 0; i < iterations; i++) {
            value = sinf(value) * cosf(value);
            value = sqrtf(fabsf(value) + 0.001f);
            value = expf(-value) * logf(value + 1.0f);
        }
        data[idx] = value;
    }
}

int main() {
    // Check CUDA device
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);
    
    // Start temperature monitoring thread
    std::thread temp_thread(monitor_gpu_stats, device);
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Let monitoring initialize
    
    float initial_temp = current_temp.load();
    printf("Initial GPU Temperature: %.0f°C\n", initial_temp);
    printf("Starting neural network training...\n\n");
    
    // Neural network setup
    float h_x[2] = {0.05f, 0.1f};
    float h_y[1] = {0.01f};

    float h_W1[6] = {0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f};
    float h_b1[3] = {0.35f, 0.35f, 0.35f};

    float h_W2[3] = {0.40f, 0.45f, 0.50f};
    float h_b2[1] = {0.60f};

    float *d_x, *d_y, *d_W1, *d_b1, *d_W2, *d_b2, *d_h_out, *d_y_pred;

    cudaMalloc(&d_x, 2*sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMalloc(&d_W1, 6*sizeof(float));
    cudaMalloc(&d_b1, 3*sizeof(float));
    cudaMalloc(&d_W2, 3*sizeof(float));
    cudaMalloc(&d_b2, sizeof(float));
    cudaMalloc(&d_h_out, 3*sizeof(float));
    cudaMalloc(&d_y_pred, sizeof(float));

    cudaMemcpy(d_x, h_x, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for intensive computation (to generate heat)
    const int heat_size = 10 * 1024 * 1024; // 10M elements
    float *d_heat_data;
    cudaMalloc(&d_heat_data, heat_size * sizeof(float));
    
    // Initialize with random data
    float *h_heat_data = (float*)malloc(heat_size * sizeof(float));
    for (int i = 0; i < heat_size; i++) {
        h_heat_data[i] = (float)rand() / RAND_MAX;
    }
    cudaMemcpy(d_heat_data, h_heat_data, heat_size * sizeof(float), cudaMemcpyHostToDevice);

    // Training loop with temperature monitoring
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Run neural network training
        forward<<<1, 3>>>(d_x, d_W1, d_b1, d_W2, d_b2, d_h_out, d_y_pred);
        backward<<<1, 3>>>(d_x, d_y, d_h_out, d_y_pred, d_W1, d_b1, d_W2, d_b2);
        
        // Run intensive computation to generate heat
        int blockSize = 256;
        int gridSize = (heat_size + blockSize - 1) / blockSize;
        intensive_computation<<<gridSize, blockSize>>>(d_heat_data, heat_size, 100);
        
        cudaDeviceSynchronize();

        // Print every epoch
        float pred;
        cudaMemcpy(&pred, d_y_pred, sizeof(float), cudaMemcpyDeviceToHost);
        float loss = 0.5f * (pred - h_y[0]) * (pred - h_y[0]);
        
        float temp = current_temp.load();
        float power = power_usage.load() / 1000.0f; // Convert to watts
        
        printf("Epoch %4d - Loss: %.8f - Predicted: %.6f | Temp: %.0f°C | Power: %.1fW\n", 
               epoch, loss, pred, temp, power);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Final results
    float final_pred;
    cudaMemcpy(&final_pred, d_y_pred, sizeof(float), cudaMemcpyDeviceToHost);
    float final_loss = 0.5f * (final_pred - h_y[0]) * (final_pred - h_y[0]);
    
    printf("\n========== Training Complete ==========\n");
    printf("Total training time: %.3f seconds\n", milliseconds / 1000.0f);
    printf("Final loss: %.8f\n", final_loss);
    printf("Final prediction: %.6f (target: %.6f)\n", final_pred, h_y[0]);
    
    printf("\n========== Temperature Stats ==========\n");
    printf("Initial temperature: %.0f°C\n", initial_temp);
    printf("Current temperature: %.0f°C\n", current_temp.load());
    printf("Peak temperature: %.0f°C\n", max_temp.load());
    printf("Temperature increase: %.0f°C\n", max_temp.load() - initial_temp);
    
    // Stop monitoring
    keep_monitoring = false;
    temp_thread.join();
    
    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h_out); cudaFree(d_y_pred);
    cudaFree(d_heat_data);
    free(h_heat_data);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// Compile with:
// nvcc -o cuda_nn_temp cuda_nn_temp.cu -lnvidia-ml -std=c++11 -lpthread
//
// If NVML is not available, you can monitor externally:
// watch -n 0.5 'nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv'
