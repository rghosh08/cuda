#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <omp.h>
#include <memory>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    int size;

    Tensor() : size(0) {}
    
    Tensor(const std::vector<int>& s) : shape(s) {
        size = 1;
        for (int dim : shape) size *= dim;
        data.resize(size, 0.0f);
    }

    float& operator()(int n, int c, int h, int w) {
        return data[n * shape[1] * shape[2] * shape[3] + 
                   c * shape[2] * shape[3] + 
                   h * shape[3] + w];
    }

    const float& operator()(int n, int c, int h, int w) const {
        return data[n * shape[1] * shape[2] * shape[3] + 
                   c * shape[2] * shape[3] + 
                   h * shape[3] + w];
    }

    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void random_init(float mean = 0.0f, float std = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);
        
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
};

class Conv2D {
private:
    Tensor weights;  // [out_channels, in_channels, kernel_h, kernel_w]
    Tensor bias;     // [out_channels]
    int in_channels, out_channels, kernel_size, stride, padding;

public:
    Conv2D(int in_ch, int out_ch, int k_size, int str = 1, int pad = 0) 
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(str), padding(pad) {
        weights = Tensor({out_ch, in_ch, k_size, k_size});
        bias = Tensor({out_ch});
        
        // Xavier initialization
        float std = std::sqrt(2.0f / (in_ch * k_size * k_size));
        weights.random_init(0.0f, std);
        bias.zero();
    }

    Tensor forward(const Tensor& input) {
        int batch_size = input.shape[0];
        int in_h = input.shape[2];
        int in_w = input.shape[3];
        
        int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
        
        Tensor output({batch_size, out_channels, out_h, out_w});
        
        #pragma omp parallel for collapse(4)
        for (int n = 0; n < batch_size; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float sum = bias.data[oc];
                        
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        sum += input(n, ic, ih, iw) * 
                                               weights.data[oc * in_channels * kernel_size * kernel_size +
                                                          ic * kernel_size * kernel_size +
                                                          kh * kernel_size + kw];
                                    }
                                }
                            }
                        }
                        output(n, oc, oh, ow) = sum;
                    }
                }
            }
        }
        return output;
    }
};

class BatchNorm2D {
private:
    Tensor gamma, beta, running_mean, running_var;
    int num_features;
    float eps;
    bool training;

public:
    BatchNorm2D(int features, float epsilon = 1e-5f) 
        : num_features(features), eps(epsilon), training(true) {
        gamma = Tensor({features});
        beta = Tensor({features});
        running_mean = Tensor({features});
        running_var = Tensor({features});
        
        std::fill(gamma.data.begin(), gamma.data.end(), 1.0f);
        beta.zero();
        running_mean.zero();
        std::fill(running_var.data.begin(), running_var.data.end(), 1.0f);
    }

    Tensor forward(const Tensor& input) {
        int batch_size = input.shape[0];
        int height = input.shape[2];
        int width = input.shape[3];
        
        Tensor output = input;
        
        if (training) {
            // Calculate batch statistics
            std::vector<float> mean(num_features, 0.0f);
            std::vector<float> var(num_features, 0.0f);
            
            #pragma omp parallel for
            for (int c = 0; c < num_features; ++c) {
                float sum = 0.0f;
                int count = batch_size * height * width;
                
                for (int n = 0; n < batch_size; ++n) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            sum += input(n, c, h, w);
                        }
                    }
                }
                mean[c] = sum / count;
                
                float sum_sq = 0.0f;
                for (int n = 0; n < batch_size; ++n) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float diff = input(n, c, h, w) - mean[c];
                            sum_sq += diff * diff;
                        }
                    }
                }
                var[c] = sum_sq / count;
            }
            
            // Normalize and apply scale/shift
            #pragma omp parallel for collapse(4)
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < num_features; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float normalized = (input(n, c, h, w) - mean[c]) / std::sqrt(var[c] + eps);
                            output(n, c, h, w) = gamma.data[c] * normalized + beta.data[c];
                        }
                    }
                }
            }
        } else {
            // Use running statistics
            #pragma omp parallel for collapse(4)
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < num_features; ++c) {
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            float normalized = (input(n, c, h, w) - running_mean.data[c]) / 
                                             std::sqrt(running_var.data[c] + eps);
                            output(n, c, h, w) = gamma.data[c] * normalized + beta.data[c];
                        }
                    }
                }
            }
        }
        
        return output;
    }

    void set_training(bool train) { training = train; }
};

class ReLU {
public:
    Tensor forward(const Tensor& input) {
        Tensor output = input;
        
        #pragma omp parallel for
        for (int i = 0; i < input.size; ++i) {
            output.data[i] = std::max(0.0f, input.data[i]);
        }
        
        return output;
    }
};

class MaxPool2D {
private:
    int kernel_size, stride, padding;

public:
    MaxPool2D(int k_size, int str = 2, int pad = 0) 
        : kernel_size(k_size), stride(str), padding(pad) {}

    Tensor forward(const Tensor& input) {
        int batch_size = input.shape[0];
        int channels = input.shape[1];
        int in_h = input.shape[2];
        int in_w = input.shape[3];
        
        int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
        int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
        
        Tensor output({batch_size, channels, out_h, out_w});
        
        #pragma omp parallel for collapse(4)
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    max_val = std::max(max_val, input(n, c, ih, iw));
                                }
                            }
                        }
                        output(n, c, oh, ow) = max_val;
                    }
                }
            }
        }
        return output;
    }
};

class AdaptiveAvgPool2D {
public:
    Tensor forward(const Tensor& input) {
        int batch_size = input.shape[0];
        int channels = input.shape[1];
        int in_h = input.shape[2];
        int in_w = input.shape[3];
        
        Tensor output({batch_size, channels, 1, 1});
        
        #pragma omp parallel for collapse(2)
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int h = 0; h < in_h; ++h) {
                    for (int w = 0; w < in_w; ++w) {
                        sum += input(n, c, h, w);
                    }
                }
                output(n, c, 0, 0) = sum / (in_h * in_w);
            }
        }
        return output;
    }
};

class Linear {
private:
    Tensor weights;  // [out_features, in_features]
    Tensor bias;     // [out_features]
    int in_features, out_features;

public:
    Linear(int in_feat, int out_feat) 
        : in_features(in_feat), out_features(out_feat) {
        weights = Tensor({out_feat, in_feat});
        bias = Tensor({out_feat});
        
        float std = std::sqrt(2.0f / in_feat);
        weights.random_init(0.0f, std);
        bias.zero();
    }

    Tensor forward(const Tensor& input) {
        int batch_size = input.shape[0];
        Tensor output({batch_size, out_features});
        
        #pragma omp parallel for collapse(2)
        for (int n = 0; n < batch_size; ++n) {
            for (int out = 0; out < out_features; ++out) {
                float sum = bias.data[out];
                for (int in = 0; in < in_features; ++in) {
                    sum += input.data[n * in_features + in] * 
                           weights.data[out * in_features + in];
                }
                output.data[n * out_features + out] = sum;
            }
        }
        return output;
    }
};

class BasicBlock {
private:
    Conv2D conv1, conv2;
    BatchNorm2D bn1, bn2;
    ReLU relu;
    std::unique_ptr<Conv2D> downsample_conv;
    std::unique_ptr<BatchNorm2D> downsample_bn;

public:
    BasicBlock(int in_channels, int out_channels, int stride = 1) 
        : conv1(in_channels, out_channels, 3, stride, 1),
          conv2(out_channels, out_channels, 3, 1, 1),
          bn1(out_channels),
          bn2(out_channels) {
        
        if (stride != 1 || in_channels != out_channels) {
            downsample_conv = std::make_unique<Conv2D>(in_channels, out_channels, 1, stride, 0);
            downsample_bn = std::make_unique<BatchNorm2D>(out_channels);
        }
    }

    Tensor forward(const Tensor& input) {
        Tensor identity = input;
        
        Tensor out = conv1.forward(input);
        out = bn1.forward(out);
        out = relu.forward(out);
        
        out = conv2.forward(out);
        out = bn2.forward(out);
        
        if (downsample_conv) {
            identity = downsample_conv->forward(input);
            identity = downsample_bn->forward(identity);
        }
        
        // Add residual connection
        #pragma omp parallel for
        for (int i = 0; i < out.size; ++i) {
            out.data[i] += identity.data[i];
        }
        
        return relu.forward(out);
    }
};

class ResNet {
private:
    Conv2D conv1;
    BatchNorm2D bn1;
    ReLU relu;
    MaxPool2D maxpool;
    
    std::vector<std::unique_ptr<BasicBlock>> layer1, layer2, layer3, layer4;
    AdaptiveAvgPool2D avgpool;
    Linear fc;

public:
    ResNet(int num_classes = 1000) 
        : conv1(3, 64, 7, 2, 3),
          bn1(64),
          maxpool(3, 2, 1),
          avgpool(),
          fc(512, num_classes) {
        
        // Layer 1: 2 blocks, 64 channels
        layer1.push_back(std::make_unique<BasicBlock>(64, 64, 1));
        layer1.push_back(std::make_unique<BasicBlock>(64, 64, 1));
        
        // Layer 2: 2 blocks, 128 channels, stride=2 for first block
        layer2.push_back(std::make_unique<BasicBlock>(64, 128, 2));
        layer2.push_back(std::make_unique<BasicBlock>(128, 128, 1));
        
        // Layer 3: 2 blocks, 256 channels, stride=2 for first block
        layer3.push_back(std::make_unique<BasicBlock>(128, 256, 2));
        layer3.push_back(std::make_unique<BasicBlock>(256, 256, 1));
        
        // Layer 4: 2 blocks, 512 channels, stride=2 for first block
        layer4.push_back(std::make_unique<BasicBlock>(256, 512, 2));
        layer4.push_back(std::make_unique<BasicBlock>(512, 512, 1));
    }

    Tensor forward(const Tensor& input) {
        Tensor x = conv1.forward(input);
        x = bn1.forward(x);
        x = relu.forward(x);
        x = maxpool.forward(x);
        
        // Apply residual layers
        for (auto& block : layer1) {
            x = block->forward(x);
        }
        for (auto& block : layer2) {
            x = block->forward(x);
        }
        for (auto& block : layer3) {
            x = block->forward(x);
        }
        for (auto& block : layer4) {
            x = block->forward(x);
        }
        
        x = avgpool.forward(x);
        
        // Flatten for fully connected layer
        Tensor flattened({x.shape[0], 512});
        #pragma omp parallel for
        for (int n = 0; n < x.shape[0]; ++n) {
            for (int c = 0; c < 512; ++c) {
                flattened.data[n * 512 + c] = x(n, c, 0, 0);
            }
        }
        
        return fc.forward(flattened);
    }

    void set_training(bool training) {
        bn1.set_training(training);
        for (auto& block : layer1) {
            // Note: In a full implementation, you'd need to expose BatchNorm layers
            // from BasicBlock to set their training mode
        }
        // Similar for other layers...
    }
};

// Example usage and testing
int main() {
    std::cout << "ResNet-18 Implementation with OpenMP" << std::endl;
    
    // Set number of OpenMP threads
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " OpenMP threads" << std::endl;
    
    // Create a ResNet-18 model
    ResNet model(1000);  // 1000 classes for ImageNet
    
    // Create dummy input (batch_size=2, channels=3, height=224, width=224)
    Tensor input({2, 3, 224, 224});
    input.random_init(0.0f, 1.0f);
    
    std::cout << "Input shape: [" << input.shape[0] << ", " << input.shape[1] 
              << ", " << input.shape[2] << ", " << input.shape[3] << "]" << std::endl;
    
    // Measure inference time
    double start_time = omp_get_wtime();
    Tensor output = model.forward(input);
    double end_time = omp_get_wtime();
    
    std::cout << "Output shape: [" << output.shape[0] << ", " << output.shape[1] << "]" << std::endl;
    std::cout << "Inference time: " << (end_time - start_time) * 1000 << " ms" << std::endl;
    
    // Print first few predictions for first sample
    std::cout << "First 10 predictions for sample 0: ";
    for (int i = 0; i < std::min(10, output.shape[1]); ++i) {
        std::cout << output.data[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

// Compilation command:
// g++ -fopenmp -O3 -std=c++14 resnet.cpp -o resnet
