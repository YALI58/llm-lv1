#include "kernel/attention.hpp"
#include "runtime/memory_manager.hpp"
#include "core/config.hpp"
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
    std::cout << "  PASSED" << std::endl;
}

void test_memory_manager_initialization() {
    std::cout << "Testing MemoryManager initialization..." << std::endl;
    
    MemoryConfig config;
    config.enable_paged_attention = true;
    config.page_size = 16;
    
    MemoryManager manager(config);
    
    manager.initialize(1024 * 1024 * 100); // 100MB
    
    assert(manager.is_initialized());
    
    auto stats = manager.get_stats();
    std::cout << "  Total memory: " << stats.total_memory << " bytes" << std::endl;
    std::cout << "  Num blocks: " << stats.num_blocks << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void test_config_validation() {
    std::cout << "Testing config validation..." << std::endl;
    
    EngineConfig config;
    config.model_path = "/test/model";
    
    bool valid = config.validate();
    assert(valid);
    
    config.memory.max_memory_usage = 1.5f; // Invalid
    valid = config.validate();
    assert(!valid);
    
    std::cout << "  Config validation works correctly" << std::endl;
    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== LLM-LV1 C++ Tests ===" << std::endl;
    std::cout << std::endl;
    
    test_attention_forward();
    test_memory_manager_initialization();
    test_config_validation();
    
    std::cout << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
