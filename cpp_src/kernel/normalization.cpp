#include "kernel/normalization.hpp"
#include <cmath>
#include <stdexcept>

namespace llm_lv1_lib {

void layer_norm_forward(float* input, float* output, const float* weight, const float* bias,
                        size_t batch_size, size_t hidden_size, float eps) {
    if (hidden_size == 0) throw std::invalid_argument("hidden_size 不能为 0");
    if (!input || !output || !weight || !bias) throw std::invalid_argument("指针不能为空");
    
    for (size_t b = 0; b < batch_size; ++b) {
        float* in = input + b * hidden_size;
        float* out = output + b * hidden_size;
        float mean = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) mean += in[i];
        mean /= static_cast<float>(hidden_size);
        float var = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) { float d = in[i] - mean; var += d * d; }
        var /= static_cast<float>(hidden_size);
        float scale = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < hidden_size; ++i) out[i] = (in[i] - mean) * scale * weight[i] + bias[i];
    }
}

void rms_norm_forward(float* input, float* output, const float* weight,
                      size_t batch_size, size_t hidden_size, float eps) {
    if (hidden_size == 0) throw std::invalid_argument("hidden_size 不能为 0");
    if (!input || !output || !weight) throw std::invalid_argument("指针不能为空");
    
    for (size_t b = 0; b < batch_size; ++b) {
        float* in = input + b * hidden_size;
        float* out = output + b * hidden_size;
        float rms = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) rms += in[i] * in[i];
        rms /= static_cast<float>(hidden_size);
        rms = std::sqrt(rms + eps);
        float scale = 1.0f / rms;
        for (size_t i = 0; i < hidden_size; ++i) out[i] = in[i] * scale * weight[i];
    }
}

} // namespace llm_lv1_lib
