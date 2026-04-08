#include <string>
#include <iostream>

namespace llm_lv1 {

// Logging utilities implementation
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

static LogLevel current_level = LogLevel::INFO;

void set_log_level(LogLevel level) {
    current_level = level;
}

void log_message(LogLevel level, const std::string& message) {
    if (level < current_level) {
        return;
    }
    
    const char* level_str = "";
    switch (level) {
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO: level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::ERROR: level_str = "ERROR"; break;
    }
    
    std::cerr << "[" << level_str << "] " << message << std::endl;
}

} // namespace llm_lv1
