#ifndef LLM_LV1_RUNTIME_MEMORY_MANAGER_HPP
#define LLM_LV1_RUNTIME_MEMORY_MANAGER_HPP

#include "core/config.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <mutex>

namespace llm_lv1 {

/**
 * @brief Memory usage statistics
 */
struct MemoryStats {
    size_t total_memory = 0;        // Total memory in bytes
    size_t allocated_memory = 0;    // Currently allocated memory
    size_t free_memory = 0;         // Currently free memory
    size_t num_blocks = 0;          // Total number of blocks
    size_t free_blocks = 0;         // Number of free blocks
    size_t peak_memory = 0;         // Peak memory usage
    
    /**
     * @brief Get memory utilization ratio
     */
    float utilization() const {
        if (total_memory == 0) return 0.0f;
        return static_cast<float>(allocated_memory) / static_cast<float>(total_memory);
    }
};

/**
 * @brief Represents a block of memory
 */
struct MemoryBlock {
    int block_id;
    size_t size;            // Size in bytes
    bool is_free;
    std::vector<float> data;
    
    MemoryBlock() : block_id(0), size(0), is_free(true) {}
    MemoryBlock(int id, size_t sz) : block_id(id), size(sz), is_free(true) {}
};

/**
 * @brief Memory manager for efficient tensor allocation
 * 
 * This class provides memory pooling and management features including:
 * - Pre-allocated memory pools to reduce allocation overhead
 * - Paged attention support for efficient KV cache management
 * - Memory statistics tracking
 * 
 * Example usage:
 * @code
 * MemoryConfig config;
 * config.enable_paged_attention = true;
 * config.page_size = 16;
 * 
 * MemoryManager manager(config);
 * manager.initialize(16 * 1024 * 1024 * 1024); // 16GB
 * 
 * auto kv_cache = manager.allocate_kv_cache(32, 32, 128, 4096);
 * @endcode
 */
class MemoryManager {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit MemoryManager(const MemoryConfig& config = MemoryConfig());
    
    /**
     * @brief Destructor
     */
    ~MemoryManager();
    
    // Disable copy
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    /**
     * @brief Initialize the memory manager
     * @param total_memory Total available memory in bytes
     * @return true if initialization successful
     */
    bool initialize(size_t total_memory);
    
    /**
     * @brief Check if memory manager is initialized
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief Allocate memory
     * @param size Number of elements
     * @return Pointer to allocated memory, or nullptr if failed
     */
    std::vector<float>* allocate(size_t size);
    
    /**
     * @brief Free allocated memory
     * @param data Pointer to memory to free
     * @return true if successful
     */
    bool free(std::vector<float>* data);
    
    /**
     * @brief Allocate KV cache for attention
     * @param num_layers Number of transformer layers
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each head
     * @param max_seq_len Maximum sequence length
     * @return KV cache structure
     */
    struct KVCache allocate_kv_cache(
        int num_layers,
        int num_heads,
        int head_dim,
        int max_seq_len
    );
    
    /**
     * @brief Get memory statistics
     */
    MemoryStats get_stats() const;
    
    /**
     * @brief Reset memory manager state
     */
    void reset();
    
private:
    MemoryConfig config_;
    bool initialized_ = false;
    size_t total_memory_ = 0;
    size_t available_memory_ = 0;
    
    std::vector<MemoryBlock> free_blocks_;
    std::unordered_map<void*, MemoryBlock*> allocated_blocks_;
    // Use indices instead of pointers to avoid pointer invalidation issues
    std::unordered_map<void*, size_t> allocated_block_indices_;
    
    mutable std::mutex mutex_;
    MemoryStats stats_;
    
    void create_paged_memory_pool();
    void create_contiguous_memory_pool();
};

/**
 * @brief KV Cache structure
 */
struct KVCache {
    std::vector<float> key_cache;       // Key cache
    std::vector<float> value_cache;     // Value cache
    std::vector<int> page_table;        // Page table for paged attention
    int page_size = 0;                  // Size of each page
    int num_layers = 0;                 // Number of layers
    int num_heads = 0;                  // Number of heads
    int head_dim = 0;                   // Head dimension
    int max_seq_len = 0;                // Maximum sequence length
    
    KVCache() = default;
};

} // namespace llm_lv1

#endif // LLM_LV1_RUNTIME_MEMORY_MANAGER_HPP
