#include "runtime/memory_manager.hpp"
#include <iostream>
#include <cassert>

using namespace llm_lv1;

void test_memory_manager_initialization() {
    std::cout << "Testing MemoryManager initialization..." << std::endl;
    
    MemoryConfig config;
    config.enable_paged_attention = true;
    config.page_size = 16;
    
    MemoryManager manager(config);
    
    bool result = manager.initialize(1024 * 1024 * 100); // 100MB
    
    assert(result);
    assert(manager.is_initialized());
    
    auto stats = manager.get_stats();
    std::cout << "  Total memory: " << stats.total_memory << " bytes" << std::endl;
    std::cout << "  Num blocks: " << stats.num_blocks << std::endl;
    std::cout << "  PASSED" << std::endl;
}

void test_kv_cache_allocation() {
    std::cout << "Testing KV cache allocation..." << std::endl;
    
    MemoryConfig config;
    config.enable_paged_attention = true;
    config.page_size = 16;
    
    MemoryManager manager(config);
    manager.initialize(1024 * 1024 * 500); // 500MB
    
    auto kv_cache = manager.allocate_kv_cache(
        4,   // num_layers
        8,   // num_heads
        64,  // head_dim
        256  // max_seq_len
    );
    
    assert(!kv_cache.key_cache.empty());
    assert(!kv_cache.value_cache.empty());
    assert(!kv_cache.page_table.empty());
    
    std::cout << "  Key cache size: " << kv_cache.key_cache.size() << std::endl;
    std::cout << "  Value cache size: " << kv_cache.value_cache.size() << std::endl;
    std::cout << "  Page table size: " << kv_cache.page_table.size() << std::endl;
    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Memory Manager Tests ===" << std::endl;
    
    test_memory_manager_initialization();
    test_kv_cache_allocation();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
