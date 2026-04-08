#include "core/config.hpp"
#include <algorithm>
#include <sstream>

namespace llm_lv1 {

EngineConfig EngineConfig::from_dict(const std::unordered_map<std::string, std::string>& params) {
    EngineConfig config;
    
    auto it = params.find("device_type");
    if (it != params.end()) {
        config.device_type = device_type_from_string(it->second);
    }
    
    it = params.find("device_id");
    if (it != params.end()) {
        try {
            config.device_id = std::stoi(it->second);
        } catch (...) {}
    }
    
    it = params.find("model_path");
    if (it != params.end()) {
        config.model_path = it->second;
    }
    
    it = params.find("quantization");
    if (it != params.end()) {
        if (it->second == "none") config.quantization = QuantizationType::NONE;
        else if (it->second == "int8") config.quantization = QuantizationType::INT8;
        else if (it->second == "int4") config.quantization = QuantizationType::INT4;
        else if (it->second == "fp8") config.quantization = QuantizationType::FP8;
        else if (it->second == "awq") config.quantization = QuantizationType::AWQ;
        else if (it->second == "gptq") config.quantization = QuantizationType::GPTQ;
    }
    
    it = params.find("max_memory_usage");
    if (it != params.end()) {
        try {
            config.memory.max_memory_usage = std::stof(it->second);
        } catch (...) {}
    }
    
    it = params.find("enable_paged_attention");
    if (it != params.end()) {
        config.memory.enable_paged_attention = (it->second == "true" || it->second == "1");
    }
    
    it = params.find("page_size");
    if (it != params.end()) {
        try {
            config.memory.page_size = std::stoi(it->second);
        } catch (...) {}
    }
    
    it = params.find("num_threads");
    if (it != params.end()) {
        try {
            config.execution.num_threads = std::stoi(it->second);
        } catch (...) {}
    }
    
    it = params.find("batch_size");
    if (it != params.end()) {
        try {
            config.execution.batch_size = std::stoi(it->second);
        } catch (...) {}
    }
    
    it = params.find("max_sequence_length");
    if (it != params.end()) {
        try {
            config.execution.max_sequence_length = std::stoi(it->second);
        } catch (...) {}
    }
    
    it = params.find("debug");
    if (it != params.end()) {
        config.debug = (it->second == "true" || it->second == "1");
    }
    
    it = params.find("log_level");
    if (it != params.end()) {
        config.log_level = it->second;
    }
    
    // Store any extra parameters
    for (const auto& kv : params) {
        const std::string& key = kv.first;
        // Skip known keys
        static const std::vector<std::string> known_keys = {
            "device_type", "device_id", "model_path", "quantization",
            "max_memory_usage", "enable_paged_attention", "page_size",
            "num_threads", "batch_size", "max_sequence_length",
            "debug", "log_level"
        };
        if (std::find(known_keys.begin(), known_keys.end(), key) == known_keys.end()) {
            config.extra_params[key] = kv.second;
        }
    }
    
    return config;
}

std::unordered_map<std::string, std::string> EngineConfig::to_dict() const {
    std::unordered_map<std::string, std::string> result;
    
    result["device_type"] = device_type_to_string(device_type);
    result["device_id"] = std::to_string(device_id);
    result["model_path"] = model_path;
    result["quantization"] = quantization_type_to_string(quantization);
    
    result["max_memory_usage"] = std::to_string(memory.max_memory_usage);
    result["enable_paged_attention"] = memory.enable_paged_attention ? "true" : "false";
    result["page_size"] = std::to_string(memory.page_size);
    
    result["num_threads"] = std::to_string(execution.num_threads);
    result["batch_size"] = std::to_string(execution.batch_size);
    result["max_sequence_length"] = std::to_string(execution.max_sequence_length);
    
    result["debug"] = debug ? "true" : "false";
    result["log_level"] = log_level;
    
    // Add extra parameters
    for (const auto& kv : extra_params) {
        result[kv.first] = kv.second;
    }
    
    return result;
}

bool EngineConfig::validate() const {
    // Check model path
    if (model_path.empty()) {
        return false;
    }
    
    // Check memory usage ratio
    if (memory.max_memory_usage <= 0.0f || memory.max_memory_usage > 1.0f) {
        return false;
    }
    
    // Check page size
    if (memory.page_size <= 0) {
        return false;
    }
    
    // Check batch size
    if (execution.batch_size <= 0) {
        return false;
    }
    
    // Check sequence length
    if (execution.max_sequence_length <= 0) {
        return false;
    }
    
    return true;
}

} // namespace llm_lv1
