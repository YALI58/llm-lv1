#include "core/engine.hpp"
#include "core/config.hpp"
#include <iostream>

using namespace llm_lv1;

int main(int argc, char* argv[]) {
    std::cout << "=== LLM-LV1 C++ Inference Engine ===" << std::endl;
    std::cout << "Simple Inference Example" << std::endl;
    std::cout << std::endl;
    
    // Create configuration
    EngineConfig config;
    config.device_type = DeviceType::CPU;
    config.model_path = "/path/to/model";  // Placeholder
    config.memory.enable_paged_attention = true;
    config.execution.max_sequence_length = 2048;
    
    // Create engine
    InferenceEngine engine(config);
    
    // Load model (will fail with placeholder path, but demonstrates API)
    std::cout << "Attempting to load model..." << std::endl;
    
    // For demonstration, we'll show the API without actually loading
    std::cout << "Engine configuration:" << std::endl;
    std::cout << "  Device: " << device_type_to_string(config.device_type) << std::endl;
    std::cout << "  Model path: " << config.model_path << std::endl;
    std::cout << "  Paged attention: " << (config.memory.enable_paged_attention ? "enabled" : "disabled") << std::endl;
    std::cout << "  Max sequence length: " << config.execution.max_sequence_length << std::endl;
    std::cout << std::endl;
    
    // Show stats
    auto stats = engine.get_stats();
    std::cout << "Engine stats:" << std::endl;
    for (const auto& kv : stats) {
        std::cout << "  " << kv.first << ": " << kv.second << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Note: This is a demonstration of the C++ API." << std::endl;
    std::cout << "To run actual inference, provide a valid model path." << std::endl;
    
    return 0;
}
