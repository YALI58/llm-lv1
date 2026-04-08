#include "core/engine.hpp"
#include "core/config.hpp"
#include <iostream>
#include <chrono>
#include <thread>

using namespace llm_lv1;

int main() {
    std::cout << "=== Streaming Generation Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create configuration
    EngineConfig config;
    config.device_type = DeviceType::CPU;
    
    // Create engine
    InferenceEngine engine(config);
    
    // Demonstrate streaming callback
    std::cout << "Simulating streaming output:" << std::endl;
    std::cout << "Prompt: Tell me a story about..." << std::endl;
    std::cout << "Response: ";
    std::cout.flush();
    
    // Simulate streaming tokens
    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "[token_" << i << "] ";
        std::cout.flush();
    }
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Streaming complete!" << std::endl;
    
    return 0;
}
