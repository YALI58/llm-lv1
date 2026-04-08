#include "core/state.hpp"
#include <algorithm>

namespace llm_lv1 {

bool StoppingCriteria::should_stop(const std::vector<int>& generated_tokens,
                                    const std::string& generated_text) const {
    // Check max tokens
    if (static_cast<int>(generated_tokens.size()) >= max_new_tokens) {
        return true;
    }
    
    // Check min tokens (don't stop if we haven't generated enough)
    if (static_cast<int>(generated_tokens.size()) < min_new_tokens) {
        return false;
    }
    
    // Check stop strings
    for (const auto& stop_str : stop_strings) {
        if (!stop_str.empty() && generated_text.find(stop_str) != std::string::npos) {
            return true;
        }
    }
    
    // Check stop token IDs
    if (!generated_tokens.empty() && !stop_token_ids.empty()) {
        int last_token = generated_tokens.back();
        if (std::find(stop_token_ids.begin(), stop_token_ids.end(), last_token) != stop_token_ids.end()) {
            return true;
        }
    }
    
    // Check EOS token
    if (eos_token_id >= 0 && !generated_tokens.empty()) {
        if (generated_tokens.back() == eos_token_id) {
            return true;
        }
    }
    
    return false;
}

GenerationState::GenerationState(const std::string& req_id)
    : request_id(req_id) {}

std::vector<int> GenerationState::get_all_tokens() const {
    std::vector<int> all_tokens;
    all_tokens.reserve(prompt_tokens.size() + generated_tokens.size());
    all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.insert(all_tokens.end(), generated_tokens.begin(), generated_tokens.end());
    return all_tokens;
}

void GenerationState::append_token(int token_id, float logprob) {
    TokenInfo info(token_id, logprob, current_position, false);
    generated_tokens.push_back(token_id);
    token_infos.push_back(info);
    current_position++;
    total_sequence_length++;
}

void GenerationState::mark_finished(const std::string& reason) {
    is_finished = true;
    finish_reason = reason;
}

void GenerationState::reset() {
    generated_tokens.clear();
    token_infos.clear();
    current_position = 0;
    total_sequence_length = 0;
    is_finished = false;
    finish_reason.clear();
    logits_processor_state.clear();
    // Keep attention_cache for potential reuse
}

std::string GenerationState::get_generated_text(void* tokenizer, bool skip_prompt) const {
    // Note: In a real implementation, this would use the tokenizer to decode tokens
    // For now, we return a placeholder
    if (tokenizer == nullptr) {
        // Return token IDs as string representation
        const std::vector<int>& tokens = skip_prompt ? generated_tokens : get_all_tokens();
        std::string result = "[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(tokens[i]);
        }
        result += "]";
        return result;
    }
    
    // With tokenizer, we would call the tokenizer's decode method
    // This is a placeholder - actual implementation depends on tokenizer library
    return "[Tokenized text - requires tokenizer implementation]";
}

} // namespace llm_lv1
