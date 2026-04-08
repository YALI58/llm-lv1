#include "kernel/attention.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace llm_lv1;

void test_attention_forward() {
    std::cout << "Testing attention_forward..." << std::endl;
    
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    int head_dim = 16;
    
    size_t total_size = batch_size * num_heads * seq_len * head_dim;
    
    std::vector<float> query(total_size, 0.5f);
    std::vector<float> key(total_size, 0.3f);
    std::vector<float> value(total_size, 0.7f);
    
    auto output = attention_forward(query, key, value, 
                                     batch_size, num_heads, seq_len, head_dim);
    
    assert(output.size() == total_size);
    std::cout << "  Output size: " << output.size() << std::endl;
    std::cout << "  First few values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void test_softmax_numerical_stability() {
    std::cout << "Testing softmax numerical stability..." << std::endl;
    
    // Create large values to test numerical stability
    int batch_size = 1;
    int num_heads = 1;
    int seq_len = 4;
    int head_dim = 8;
    
    size_t total_size = batch_size * num_heads * seq_len * head_dim;
    
    std::vector<float> query(total_size, 100.0f);  // Large values
    std::vector<float> key(total_size, 100.0f);
    std::vector<float> value(total_size, 1.0f);
    
    auto output = attention_forward(query, key, value,
                                     batch_size, num_heads, seq_len, head_dim);
    
    // Check that output doesn't contain NaN or Inf
    bool valid = true;
    for (float v : output) {
        if (std::isnan(v) || std::isinf(v)) {
            valid = false;
            break;
        }
    }
    
    assert(valid);
    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Attention Kernel Tests ===" << std::endl;
    
    test_attention_forward();
    test_softmax_numerical_stability();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
