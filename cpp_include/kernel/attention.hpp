#ifndef LLM_LV1_KERNEL_ATTENTION_HPP
#define LLM_LV1_KERNEL_ATTENTION_HPP

#include <vector>
#include <cstdint>
#include <memory>

namespace llm_lv1 {

/**
 * @brief Compute scaled dot-product attention
 * 
 * This function implements the standard attention mechanism:
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
 * 
 * @param query Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param key Key tensor [batch_size, num_heads, seq_len, head_dim]
 * @param value Value tensor [batch_size, num_heads, seq_len, head_dim]
 * @param scale Optional scaling factor (default: 1/sqrt(head_dim))
 * @param mask Optional attention mask
 * @return Output tensor [batch_size, num_heads, seq_len, head_dim]
 */
std::vector<float> attention_forward(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale = -1.0f,
    const std::vector<float>* mask = nullptr
);

/**
 * @brief Paged attention with virtual memory management
 * 
 * This implements attention with paged KV cache, similar to vLLM's approach.
 * The KV cache is stored in non-contiguous memory pages, and the page table
 * maps logical token positions to physical page locations.
 * 
 * @param query Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param key_cache Paged key cache [num_layers, num_pages, num_heads, page_size, head_dim]
 * @param value_cache Paged value cache (same shape as key_cache)
 * @param page_table Page table mapping logical to physical pages
 * @param context_lengths Actual sequence lengths for each batch item
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param page_size Size of each page (tokens per page)
 * @param scale Optional scaling factor
 * @return Output tensor [batch_size, num_heads, seq_len, head_dim]
 */
std::vector<float> paged_attention_forward(
    const std::vector<float>& query,
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache,
    const std::vector<int>& page_table,
    const std::vector<int>& context_lengths,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int page_size,
    int num_pages,
    float scale = -1.0f
);

/**
 * @brief Flash attention optimization (placeholder for future implementation)
 * 
 * This will be implemented using CUDA kernels for efficient attention computation.
 */
class FlashAttention {
public:
    FlashAttention() = default;
    
    /**
     * @brief Forward pass with flash attention
     */
    std::vector<float> forward(
        const std::vector<float>& query,
        const std::vector<float>& key,
        const std::vector<float>& value,
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        float scale = -1.0f
    );
};

/**
 * @brief Multi-head attention wrapper
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(int num_heads, int head_dim, float scale = -1.0f);
    
    /**
     * @brief Forward pass
     */
    std::vector<float> forward(
        const std::vector<float>& query,
        const std::vector<float>& key,
        const std::vector<float>& value,
        int batch_size,
        int seq_len,
        const std::vector<float>* mask = nullptr
    );
    
    /**
     * @brief Update KV cache
     */
    void update_kv_cache(
        const std::vector<float>& key,
        const std::vector<float>& value,
        int batch_size,
        int seq_len,
        int position
    );
    
private:
    int num_heads_;
    int head_dim_;
    float scale_;
    std::vector<float> kv_cache_;
    bool has_cache_ = false;
};

} // namespace llm_lv1

#endif // LLM_LV1_KERNEL_ATTENTION_HPP
