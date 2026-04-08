#ifndef LLM_LV1_CORE_ENGINE_HPP
#define LLM_LV1_CORE_ENGINE_HPP

#include "core/config.hpp"
#include "core/state.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace llm_lv1 {

// Forward declarations
class MemoryManager;
class Tokenizer;

/**
 * @brief Main inference engine for LLM text generation
 * 
 * This engine provides a unified interface for loading models and
 * generating text with various sampling strategies. It supports
 * both synchronous and asynchronous generation, as well as
 * streaming output.
 * 
 * Example usage:
 * @code
 * EngineConfig config;
 * config.device_type = DeviceType::CUDA;
 * config.model_path = "/path/to/model";
 * 
 * InferenceEngine engine(config);
 * 
 * if (engine.load_model()) {
 *     std::string result = engine.generate("Hello, how are you?");
 *     std::cout << result << std::endl;
 *     
 *     // Streaming generation
 *     engine.generate_stream("Tell me a story", [](const std::string& chunk) {
 *         std::cout << chunk << std::flush;
 *     });
 * }
 * @endcode
 */
class InferenceEngine {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit InferenceEngine(const EngineConfig& config = EngineConfig());
    
    /**
     * @brief Destructor
     */
    ~InferenceEngine();
    
    // Disable copy
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    /**
     * @brief Load the model and tokenizer
     * @param model_path Path to model files (optional, uses config if not provided)
     * @param tokenizer_path Path to tokenizer files (optional, uses model_path if not provided)
     * @return true if loading successful
     */
    bool load_model(const std::string& model_path = "", const std::string& tokenizer_path = "");
    
    /**
     * @brief Unload the model and free resources
     */
    void unload_model();
    
    /**
     * @brief Check if model is loaded
     */
    bool is_loaded() const { return is_loaded_; }
    
    /**
     * @brief Get engine configuration
     */
    const EngineConfig& get_config() const { return config_; }
    
    /**
     * @brief Generate text synchronously
     * @param prompt Input prompt text
     * @param stopping_criteria Criteria for stopping generation
     * @return Generated text string
     */
    std::string generate(
        const std::string& prompt,
        const StoppingCriteria& stopping_criteria = StoppingCriteria()
    );
    
    /**
     * @brief Generate text with streaming output
     * @param prompt Input prompt text
     * @param callback Callback function called for each generated chunk
     * @param stopping_criteria Criteria for stopping generation
     */
    void generate_stream(
        const std::string& prompt,
        const std::function<void(const std::string&)>& callback,
        const StoppingCriteria& stopping_criteria = StoppingCriteria()
    );
    
    /**
     * @brief Tokenize input text
     * @param text Input text to tokenize
     * @return List of token IDs
     */
    std::vector<int> tokenize(const std::string& text);
    
    /**
     * @brief Convert token IDs back to text
     * @param tokens List of token IDs
     * @return Decoded text string
     */
    std::string detokenize(const std::vector<int>& tokens);
    
    /**
     * @brief Get engine statistics
     */
    std::unordered_map<std::string, std::string> get_stats() const;
    
private:
    EngineConfig config_;
    bool is_loaded_ = false;
    
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    std::unordered_map<std::string, std::shared_ptr<GenerationState>> active_states_;
    
    void setup_logging();
    void ensure_loaded() const;
    
    std::shared_ptr<GenerationState> create_generation_state(
        const std::vector<int>& prompt_tokens,
        const StoppingCriteria& stopping_criteria
    );
    
    void remove_generation_state(const std::string& request_id);
    
    void run_generation(
        std::shared_ptr<GenerationState> state,
        const StoppingCriteria& stopping_criteria
    );
    
    void run_generation_stream(
        std::shared_ptr<GenerationState> state,
        const StoppingCriteria& stopping_criteria,
        const std::function<void(const std::string&)>& callback
    );
};

} // namespace llm_lv1

#endif // LLM_LV1_CORE_ENGINE_HPP
