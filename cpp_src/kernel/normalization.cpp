#include <vector>
#include <cmath>

namespace llm_lv1 {

/**
 * @brief Layer normalization forward pass
 */
std::vector<float> layer_norm_forward(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int hidden_size,
    float eps = 1e-5f
) {
    std::vector<float> output(input.size());
    
    size_t num_elements = input.size();
    size_t num_rows = num_elements / hidden_size;
    
    for (size_t row = 0; row < num_rows; ++row) {
        size_t base = row * hidden_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            mean += input[base + i];
        }
        mean /= hidden_size;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float diff = input[base + i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;
        
        // Normalize and apply scale/shift
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int i = 0; i < hidden_size; ++i) {
            float normalized = (input[base + i] - mean) * inv_std;
            output[base + i] = normalized * weight[i] + bias[i];
        }
    }
    
    return output;
}

/**
 * @brief RMS normalization forward pass
 */
std::vector<float> rms_norm_forward(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int hidden_size,
    float eps = 1e-5f
) {
    std::vector<float> output(input.size());
    
    size_t num_elements = input.size();
    size_t num_rows = num_elements / hidden_size;
    
    for (size_t row = 0; row < num_rows; ++row) {
        size_t base = row * hidden_size;
        
        // Compute RMS
        float rms = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            rms += input[base + i] * input[base + i];
        }
        rms /= hidden_size;
        rms = std::sqrt(rms + eps);
        
        // Normalize and apply scale
        for (int i = 0; i < hidden_size; ++i) {
            output[base + i] = (input[base + i] / rms) * weight[i];
        }
    }
    
    return output;
}

} // namespace llm_lv1
