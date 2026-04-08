#include <vector>
#include <cmath>
#include <algorithm>

namespace llm_lv1 {

/**
 * @brief GELU activation function
 */
std::vector<float> gelu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const float sqrt_2_pi = 0.7978845608028654f;
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
    }
    
    return output;
}

/**
 * @brief SiLU/Swish activation function
 */
std::vector<float> silu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
    
    return output;
}

/**
 * @brief ReLU activation function
 */
std::vector<float> relu_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    
    return output;
}

} // namespace llm_lv1
