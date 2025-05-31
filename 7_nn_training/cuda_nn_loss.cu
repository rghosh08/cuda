#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define LR 0.5f
#define EPOCHS 1000

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

int main() {
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

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        forward<<<1, 3>>>(d_x, d_W1, d_b1, d_W2, d_b2, d_h_out, d_y_pred);
        cudaDeviceSynchronize();

        backward<<<1, 3>>>(d_x, d_y, d_h_out, d_y_pred, d_W1, d_b1, d_W2, d_b2);
        cudaDeviceSynchronize();

        if (epoch % 100 == 0 || epoch == 1) {
            float pred;
            cudaMemcpy(&pred, d_y_pred, sizeof(float), cudaMemcpyDeviceToHost);
            float loss = 0.5f * (pred - h_y[0]) * (pred - h_y[0]);
            printf("Epoch %d - Loss: %.8f - Predicted: %.6f\n", epoch, loss, pred);
        }
    }

    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h_out); cudaFree(d_y_pred);

    return 0;
}

