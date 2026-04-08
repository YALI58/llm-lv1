#ifndef LLM_LV1_CORE_CONFIG_HPP
#define LLM_LV1_CORE_CONFIG_HPP

#include <string>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llm_lv1 {

/**
 * @brief Supported device types for inference
 */
enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    WEBGPU,
    METAL
};

/**
 * @brief Convert DeviceType to string
 */
inline std::string device_type_to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::ROCM: return "rocm";
        case DeviceType::WEBGPU: return "webgpu";
        case DeviceType::METAL: return "metal";
        default: return "unknown";
    }
}

/**
 * @brief Parse DeviceType from string
 */
inline DeviceType device_type_from_string(const std::string& str) {
    if (str == "cpu") return DeviceType::CPU;
    if (str == "cuda") return DeviceType::CUDA;
    if (str == "rocm") return DeviceType::ROCM;
    if (str == "webgpu") return DeviceType::WEBGPU;
    if (str == "metal") return DeviceType::METAL;
    return DeviceType::CPU; // default
}

/**
 * @brief Supported quantization types
 */
enum class QuantizationType {
    NONE,
    INT8,
    INT4,
    FP8,
    AWQ,
    GPTQ
};

/**
 * @brief Convert QuantizationType to string
 */
inline std::string quantization_type_to_string(QuantizationType type) {
    switch (type) {
        case QuantizationType::NONE: return "none";
        case QuantizationType::INT8: return "int8";
        case QuantizationType::INT4: return "int4";
        case QuantizationType::FP8: return "fp8";
        case QuantizationType::AWQ: return "awq";
        case QuantizationType::GPTQ: return "gptq";
        default: return "unknown";
    }
}

/**
 * @brief Memory management configuration
 */
struct MemoryConfig {
    float max_memory_usage = 0.9f;        // Maximum memory usage ratio
    bool enable_paged_attention = true;   // Enable paged attention
    int page_size = 16;                   // Page size for paged attention
    int max_num_blocks_per_seq = 256;     // Max blocks per sequence
    
    MemoryConfig() = default;
};

/**
 * @brief Execution strategy configuration
 */
struct ExecutionConfig {
    int num_threads = -1;                 // -1 means auto-detect
    int batch_size = 1;                   // Default batch size
    int max_sequence_length = 4096;       // Max sequence length
    bool enable_cuda_graph = false;       // Enable CUDA graph optimization
    bool stream_mode = true;              // Enable streaming mode
    
    ExecutionConfig() = default;
};

/**
 * @brief Main engine configuration
 * 
 * This class provides configuration options for the inference engine,
 * including device settings, memory management, and execution strategies.
 * 
 * Example usage:
 * @code
 * EngineConfig config;
 * config.device_type = DeviceType::CUDA;
 * config.model_path = "/path/to/model";
 * config.memory.enable_paged_attention = true;
 * 
 * InferenceEngine engine(config);
 * @endcode
 */
class EngineConfig {
public:
    DeviceType device_type = DeviceType::CUDA;
    int device_id = 0;
    std::string model_path;
    QuantizationType quantization = QuantizationType::NONE;
    
    MemoryConfig memory;
    ExecutionConfig execution;
    
    bool debug = false;
    std::string log_level = "info";
    
    // Custom parameters for extensibility
    std::unordered_map<std::string, std::string> extra_params;
    
    /**
     * @brief Default constructor
     */
    EngineConfig() = default;
    
    /**
     * @brief Create config from string map
     */
    static EngineConfig from_dict(const std::unordered_map<std::string, std::string>& params);
    
    /**
     * @brief Convert config to string map
     */
    std::unordered_map<std::string, std::string> to_dict() const;
    
    /**
     * @brief Validate configuration
     */
    bool validate() const;
};

} // namespace llm_lv1

#endif // LLM_LV1_CORE_CONFIG_HPP
