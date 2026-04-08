#include "kernel/attention.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace llm_lv1 {

// Helper function for softmax
static void softmax(std::vector<float>& x, int start, int count) {
    if (count <= 0) return;
    
    // Find max for numerical stability
    float max_val = x[start];
    for (int i = 1; i < count; ++i) {
        max_val = std::max(max_val, x[start + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        x[start + i] = std::exp(x[start + i] - max_val);
        sum += x[start + i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < count; ++i) {
            x[start + i] /= sum;
        }
    }
}

std::vector<float> attention_forward(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    const std::vector<float>* mask
) {
    // Calculate scale if not provided
    if (scale < 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }
    
    // Output tensor
    std::vector<float> output(query.size(), 0.0f);
    
    // Temporary buffers for attention computation
    std::vector<float> scores(seq_len * seq_len);
    std::vector<float> weights(seq_len * seq_len);
    
    // For each batch and head
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int base_q = (b * num_heads + h) * seq_len * head_dim;
            int base_k = (b * num_heads + h) * seq_len * head_dim;
            int base_v = (b * num_heads + h) * seq_len * head_dim;
            int base_out = (b * num_heads + h) * seq_len * head_dim;
            
            // Compute Q @ K^T
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += query[base_q + q_pos * head_dim + d] * 
                                 key[base_k + k_pos * head_dim + d];
                    }
                    score *= scale;
                    
                    // Apply mask if provided
                    if (mask != nullptr && !mask->empty()) {
                        score += (*mask)[q_pos * seq_len + k_pos];
                    }
                    
                    scores[q_pos * seq_len + k_pos] = score;
                }
            }
            
            // Softmax over keys dimension
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                softmax(scores, q_pos * seq_len, seq_len);
                
                // Copy weights
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    weights[q_pos * seq_len + k_pos] = scores[q_pos * seq_len + k_pos];
                }
            }
            
            // Apply attention to values: weights @ V
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                for (int d = 0; d < head_dim; ++d) {
                    float out_val = 0.0f;
                    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                        out_val += weights[q_pos * seq_len + k_pos] * 
                                   value[base_v + k_pos * head_dim + d];
                    }
                    output[base_out + q_pos * head_dim + d] = out_val;
                }
            }
        }
    }
    
    return output;
}

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
    float scale
) {
    if (scale < 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }
    
    std::vector<float> output(query.size(), 0.0f);
    
    // For each sequence in the batch
    for (int b = 0; b < batch_size; ++b) {
        int context_len = context_lengths[b];
        
        // For each head
        for (int h = 0; h < num_heads; ++h) {
            int base_q = (b * num_heads + h) * seq_len * head_dim;
            
            // For each query position
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                // Gather keys and values from paged cache
                std::vector<float> keys_gathered(context_len * head_dim, 0.0f);
                std::vector<float> values_gathered(context_len * head_dim, 0.0f);
                
                for (int kv_pos = 0; kv_pos < context_len; ++kv_pos) {
                    int page_idx = kv_pos / page_size;
                    int offset_in_page = kv_pos % page_size;
                    
                    if (page_idx >= static_cast<int>(page_table.size()) / (batch_size * (seq_len / page_size + 1))) {
                        continue;
                    }
                    
                    int physical_page = page_table[b * (seq_len / page_size + 1) + page_idx];
                    if (physical_page < 0 || physical_page >= num_pages) {
                        continue;
                    }
                    
                    // Extract key and value from cache
                    int cache_base = physical_page * num_heads * page_size * head_dim + 
                                     h * page_size * head_dim + 
                                     offset_in_page * head_dim;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        keys_gathered[kv_pos * head_dim + d] = key_cache[cache_base + d];
                        values_gathered[kv_pos * head_dim + d] = value_cache[cache_base + d];
                    }
                }
                
                // Compute attention scores
                std::vector<float> attn_scores(context_len, 0.0f);
                for (int kv_pos = 0; kv_pos < context_len; ++kv_pos) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += query[base_q + q_pos * head_dim + d] * 
                                 keys_gathered[kv_pos * head_dim + d];
                    }
                    attn_scores[kv_pos] = score * scale;
                }
                
                // Softmax
                softmax(attn_scores, 0, context_len);
                
                // Apply attention to values
                for (int d = 0; d < head_dim; ++d) {
                    float out_val = 0.0f;
                    for (int kv_pos = 0; kv_pos < context_len; ++kv_pos) {
                        out_val += attn_scores[kv_pos] * values_gathered[kv_pos * head_dim + d];
                    }
                    output[base_q + q_pos * head_dim + d] = out_val;
                }
            }
        }
    }
    
    return output;
}

std::vector<float> FlashAttention::forward(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Placeholder: falls back to standard attention
    // TODO: Implement optimized CUDA kernel
    return attention_forward(query, key, value, batch_size, num_heads, seq_len, head_dim, scale);
}

MultiHeadAttention::MultiHeadAttention(int num_heads, int head_dim, float scale)
    : num_heads_(num_heads), head_dim_(head_dim) {
    scale_ = (scale < 0.0f) ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : scale;
}

std::vector<float> MultiHeadAttention::forward(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    int batch_size,
    int seq_len,
    const std::vector<float>* mask
) {
    return attention_forward(query, key, value, batch_size, num_heads_, seq_len, head_dim_, scale_, mask);
}

void MultiHeadAttention::update_kv_cache(
    const std::vector<float>& key,
    const std::vector<float>& value,
    int batch_size,
    int seq_len,
    int position
) {
    // Update KV cache with new key/value
    // This is a simplified implementation
    size_t cache_size = batch_size * num_heads_ * seq_len * head_dim_;
    
    if (kv_cache_.size() < cache_size * 2) {
        kv_cache_.resize(cache_size * 2, 0.0f);
    }
    
    // Copy new keys and values to cache
    std::memcpy(kv_cache_.data(), key.data(), key.size() * sizeof(float));
    std::memcpy(kv_cache_.data() + cache_size, value.data(), value.size() * sizeof(float));
    
    has_cache_ = true;
}

} // namespace llm_lv1
