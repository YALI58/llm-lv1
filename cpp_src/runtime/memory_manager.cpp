#include "runtime/memory_manager.hpp"
#include <algorithm>
#include <numeric>

namespace llm_lv1 {

MemoryManager::MemoryManager(const MemoryConfig& config)
    : config_(config) {}

MemoryManager::~MemoryManager() {
    reset();
}

bool MemoryManager::initialize(size_t total_memory) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true;
    }
    
    total_memory_ = total_memory;
    available_memory_ = static_cast<size_t>(total_memory * config_.max_memory_usage);
    
    if (config_.enable_paged_attention) {
        create_paged_memory_pool();
    } else {
        create_contiguous_memory_pool();
    }
    
    initialized_ = true;
    return true;
}

void MemoryManager::create_paged_memory_pool() {
    int page_size = config_.page_size;
    int max_num_blocks = config_.max_num_blocks_per_seq;
    
    // Calculate block size (simplified - in reality depends on model architecture)
    size_t block_size = page_size * 1024; // 1KB per page initially
    
    size_t num_blocks = std::min(
        available_memory_ / block_size,
        static_cast<size_t>(max_num_blocks * 10)
    );
    
    free_blocks_.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
        free_blocks_.emplace_back(static_cast<int>(i), block_size);
    }
    
    stats_.num_blocks = num_blocks;
    stats_.free_blocks = num_blocks;
    stats_.total_memory = num_blocks * block_size;
}

void MemoryManager::create_contiguous_memory_pool() {
    // For contiguous allocation, we don't pre-allocate blocks
    // Instead, we track total available memory
    stats_.total_memory = available_memory_;
    stats_.free_memory = available_memory_;
}

std::vector<float>* MemoryManager::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return nullptr;
    }
    
    size_t bytes_needed = size * sizeof(float);
    
    // Try to allocate from blocks if using paged attention
    if (config_.enable_paged_attention && !free_blocks_.empty()) {
        // Find a suitable block or combine multiple blocks
        auto it = std::find_if(free_blocks_.begin(), free_blocks_.end(),
            [bytes_needed](const MemoryBlock& block) {
                return block.size >= bytes_needed;
            });
        
        if (it != free_blocks_.end()) {
            // Store iterator before erase to avoid use-after-erase
            MemoryBlock* block_ptr = &(*it);
            it->is_free = false;
            it->data.resize(size, 0.0f);
            allocated_blocks_[it->data.data()] = block_ptr;
            
            stats_.allocated_memory += it->size;
            stats_.free_blocks--;
            stats_.peak_memory = std::max(stats_.peak_memory, stats_.allocated_memory);
            
            free_blocks_.erase(it);
            return &block_ptr->data;
        }
    }
    
    // Fallback: direct allocation
    auto* data = new std::vector<float>(size, 0.0f);
    allocated_blocks_[data->data()] = nullptr; // Mark as externally allocated
    stats_.allocated_memory += bytes_needed;
    stats_.peak_memory = std::max(stats_.peak_memory, stats_.allocated_memory);
    
    return data;
}

bool MemoryManager::free(std::vector<float>* data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || data == nullptr) {
        return false;
    }
    
    auto it = allocated_blocks_.find(data->data());
    if (it != allocated_blocks_.end()) {
        MemoryBlock* block = it->second;
        if (block != nullptr) {
            // Return block to free pool
            block->is_free = true;
            block->data.clear();
            free_blocks_.push_back(*block);
            stats_.free_blocks++;
            stats_.allocated_memory -= block->size;
        } else {
            // Externally allocated block
            stats_.allocated_memory -= data->size() * sizeof(float);
            delete data;
        }
        allocated_blocks_.erase(it);
        return true;
    }
    
    return false;
}

KVCache MemoryManager::allocate_kv_cache(
    int num_layers,
    int num_heads,
    int head_dim,
    int max_seq_len
) {
    KVCache cache;
    cache.num_layers = num_layers;
    cache.num_heads = num_heads;
    cache.head_dim = head_dim;
    cache.max_seq_len = max_seq_len;
    
    if (config_.enable_paged_attention) {
        int page_size = config_.page_size;
        int num_pages_per_layer = (max_seq_len + page_size - 1) / page_size;
        
        // Allocate paged KV cache
        // Shape: [num_layers, num_pages, num_heads, page_size, head_dim]
        size_t elements_per_layer = num_pages_per_layer * num_heads * page_size * head_dim;
        cache.key_cache.resize(elements_per_layer * num_layers, 0.0f);
        cache.value_cache.resize(elements_per_layer * num_layers, 0.0f);
        
        // Create page table
        int pages_per_seq = max_seq_len / page_size + 1;
        cache.page_table.resize(num_layers * pages_per_seq, -1);
        
        // Initialize page table with sequential mapping (simplified)
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int page = 0; page < pages_per_seq; ++page) {
                cache.page_table[layer * pages_per_seq + page] = page;
            }
        }
        
        cache.page_size = page_size;
    } else {
        // Contiguous KV cache
        // Shape: [num_layers, 2, num_heads, max_seq_len, head_dim]
        size_t elements = num_layers * 2 * num_heads * max_seq_len * head_dim;
        cache.key_cache.resize(elements, 0.0f);
        cache.value_cache.resize(elements, 0.0f);
        cache.page_size = max_seq_len;
    }
    
    return cache;
}

MemoryStats MemoryManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryStats result = stats_;
    result.free_memory = total_memory_ - result.allocated_memory;
    return result;
}

void MemoryManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Free all externally allocated blocks
    for (auto& pair : allocated_blocks_) {
        if (pair.second == nullptr) {
            // Externally allocated vector<float>* - properly delete it
            delete static_cast<std::vector<float>*>(pair.first);
        }
    }
    
    allocated_blocks_.clear();
    
    // Reset free blocks
    for (auto& block : free_blocks_) {
        block.is_free = true;
        block.data.clear();
    }
    
    stats_.allocated_memory = 0;
    stats_.free_blocks = stats_.num_blocks;
}

} // namespace llm_lv1
