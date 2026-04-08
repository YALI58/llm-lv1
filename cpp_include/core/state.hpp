#ifndef LLM_LV1_CORE_STATE_HPP
#define LLM_LV1_CORE_STATE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace llm_lv1 {

/**
 * @brief Criteria for stopping text generation
 */
struct StoppingCriteria {
    int max_new_tokens = 256;       // Maximum number of new tokens to generate
    int min_new_tokens = 0;         // Minimum number of new tokens to generate
    std::vector<std::string> stop_strings;  // Strings that trigger stopping
    std::vector<int> stop_token_ids;        // Token IDs that trigger stopping
    int eos_token_id = -1;          // End-of-sequence token ID
    float temperature = 1.0f;       // Sampling temperature
    float top_p = 0.9f;             // Nucleus sampling parameter
    int top_k = -1;                 // Top-k sampling (-1 means no limit)
    float repetition_penalty = 1.0f; // Penalty for repetition
    
    /**
     * @brief Check if generation should stop
     * @param generated_tokens List of generated token IDs
     * @param generated_text Generated text string
     * @return true if generation should stop
     */
    bool should_stop(const std::vector<int>& generated_tokens, 
                     const std::string& generated_text) const;
};

/**
 * @brief Information about a generated token
 */
struct TokenInfo {
    int token_id;           // Token ID
    float logprob;          // Log probability
    int position;           // Position in sequence
    bool is_eos;            // Whether this is an EOS token
    
    TokenInfo() : token_id(0), logprob(0.0f), position(0), is_eos(false) {}
    TokenInfo(int id, float prob, int pos, bool eos = false)
        : token_id(id), logprob(prob), position(pos), is_eos(eos) {}
};

/**
 * @brief State tracking for a single generation request
 * 
 * This class maintains the state of an ongoing text generation,
 * including generated tokens, attention cache, and completion status.
 */
class GenerationState {
public:
    std::string request_id;                 // Unique request identifier
    std::vector<int> prompt_tokens;         // Input prompt tokens
    std::vector<int> generated_tokens;      // Generated tokens
    std::vector<TokenInfo> token_infos;     // Detailed token information
    
    // Attention cache (KV cache)
    std::unordered_map<std::string, void*> attention_cache;
    
    // Position tracking
    int current_position = 0;               // Current position in generation
    int total_sequence_length = 0;          // Total sequence length
    
    // State flags
    bool is_finished = false;               // Whether generation is complete
    std::string finish_reason;              // Reason for finishing
    
    // Sampling state
    std::unordered_map<std::string, void*> logits_processor_state;
    
    /**
     * @brief Default constructor
     */
    GenerationState() = default;
    
    /**
     * @brief Constructor with request ID
     */
    explicit GenerationState(const std::string& req_id);
    
    /**
     * @brief Get all tokens (prompt + generated)
     */
    std::vector<int> get_all_tokens() const;
    
    /**
     * @brief Get number of generated tokens
     */
    size_t num_generated_tokens() const { return generated_tokens.size(); }
    
    /**
     * @brief Get total number of tokens
     */
    size_t total_tokens() const { return prompt_tokens.size() + generated_tokens.size(); }
    
    /**
     * @brief Append a generated token
     */
    void append_token(int token_id, float logprob = 0.0f);
    
    /**
     * @brief Mark generation as finished
     */
    void mark_finished(const std::string& reason);
    
    /**
     * @brief Reset state for reuse
     */
    void reset();
    
    /**
     * @brief Get generated text (requires tokenizer)
     */
    std::string get_generated_text(void* tokenizer = nullptr, bool skip_prompt = true) const;
};

} // namespace llm_lv1

#endif // LLM_LV1_CORE_STATE_HPP
