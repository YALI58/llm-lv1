#include "core/engine.hpp"
#include "core/config.hpp"
#include <iostream>
#include <cassert>

using namespace llm_lv1;

void test_engine_creation() {
    std::cout << "Testing engine creation..." << std::endl;
    
    EngineConfig config;
    config.device_type = DeviceType::CPU;
    
    InferenceEngine engine(config);
    
    assert(!engine.is_loaded());
    std::cout << "  Engine created successfully" << std::endl;
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
    std::cout << "=== Engine Tests ===" << std::endl;
    
    test_engine_creation();
    test_config_validation();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
