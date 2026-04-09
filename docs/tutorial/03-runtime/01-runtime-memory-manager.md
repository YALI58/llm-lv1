# memory_manager.hpp 完全指南 - 内存管理详解

## 📖 文件概览

**文件位置**: `cpp_include/runtime/memory_manager.hpp`

**作用**: 高效的内存管理系统，提供：
- 内存池化（减少分配开销）
- 分页注意力支持
- KV Cache 管理
- 内存统计追踪

---

## 🔍 为什么需要专门的内存管理器？

### 问题：频繁内存分配的代价

```cpp
// ❌ 低效方式
for (int i = 0; i < 1000; ++i) {
    float* tensor = new float[1024];  // 每次分配都耗时
    // ... 使用 ...
    delete[] tensor;                   // 释放也耗时
}
```

**问题**:
- `malloc/new` 和 `free/delete` 很慢
- 导致内存碎片
- 无法复用已分配内存

### 解决方案：内存池

```cpp
// ✅ 高效方式
MemoryPool pool(1024 * 1024);  // 预分配 1MB

for (int i = 0; i < 1000; ++i) {
    float* tensor = pool.allocate(1024);  // 快速分配
    // ... 使用 ...
    pool.free(tensor);                     // 快速回收
}
```

**优势**:
- 一次大分配，多次小使用
- 零碎片
- 复用内存块

---

## 🔍 代码逐行解析

### 1. MemoryStats 结构体

```cpp
struct MemoryStats {
    size_t total_memory = 0;        // 总内存（字节）
    size_t allocated_memory = 0;    // 已分配内存
    size_t free_memory = 0;         // 空闲内存
    size_t num_blocks = 0;          // 总块数
    size_t free_blocks = 0;         // 空闲块数
    size_t peak_memory = 0;         // 峰值使用量
    
    float utilization() const {
        if (total_memory == 0) return 0.0f;
        return static_cast<float>(allocated_memory) / 
               static_cast<float>(total_memory);
    }
};
```

**用途**:
- 监控内存使用情况
- 调试内存泄漏
- 性能分析

---

### 2. MemoryBlock 结构体

```cpp
struct MemoryBlock {
    int block_id;
    size_t size;            // 大小（字节）
    bool is_free;
    std::vector<float> data;
    
    MemoryBlock() : block_id(0), size(0), is_free(true) {}
    MemoryBlock(int id, size_t sz) : block_id(id), size(sz), is_free(true) {}
};
```

**说明**:
- 每个块有唯一 ID
- 记录是否空闲
- `data` 存储实际数据

---

### 3. MemoryManager 类

```cpp
class MemoryManager {
public:
    explicit MemoryManager(const MemoryConfig& config = MemoryConfig());
    ~MemoryManager();
    
    // 禁用拷贝
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    // 初始化
    bool initialize(size_t total_memory);
    bool is_initialized() const { return initialized_; }
    
    // 分配和释放
    std::vector<float>* allocate(size_t size);
    bool free(std::vector<float>* data);
    
    // KV Cache 专用分配
    struct KVCache allocate_kv_cache(
        int num_layers,
        int num_heads,
        int head_dim,
        int max_seq_len
    );
    
    // 统计信息
    MemoryStats get_stats() const;
    void reset();
    
private:
    MemoryConfig config_;
    bool initialized_ = false;
    size_t total_memory_ = 0;
    size_t available_memory_ = 0;
    
    std::vector<MemoryBlock> free_blocks_;
    std::unordered_map<void*, MemoryBlock*> allocated_blocks_;
    std::unordered_map<void*, size_t> allocated_block_indices_;
    
    mutable std::mutex mutex_;
    MemoryStats stats_;
    
    void create_paged_memory_pool();
    void create_contiguous_memory_pool();
};
```

---

## 💡 使用示例

### 示例 1: 基础内存管理

```cpp
#include "runtime/memory_manager.hpp"
#include <iostream>

using namespace llm_lv1;

int main() {
    // 1. 创建配置
    MemoryConfig config;
    config.enable_paged_attention = true;
    config.page_size = 16;
    
    // 2. 创建管理器
    MemoryManager manager(config);
    
    // 3. 初始化（分配 256MB）
    manager.initialize(256 * 1024 * 1024);
    
    // 4. 分配内存
    auto* tensor1 = manager.allocate(1024);   // 1024 floats
    auto* tensor2 = manager.allocate(2048);   // 2048 floats
    
    // 5. 使用内存
    for (size_t i = 0; i < tensor1->size(); ++i) {
        (*tensor1)[i] = static_cast<float>(i);
    }
    
    // 6. 查看统计
    MemoryStats stats = manager.get_stats();
    std::cout << "利用率：" << stats.utilization() * 100 << "%" << std::endl;
    
    // 7. 释放内存
    manager.free(tensor1);
    manager.free(tensor2);
    
    return 0;
}
```

---

### 示例 2: KV Cache 分配

```cpp
// 为 Transformer 分配 KV Cache
KVCache kv_cache = manager.allocate_kv_cache(
    32,   // num_layers: 32 层（如 LLaMA-2-7B）
    32,   // num_heads: 32 个头
    128,  // head_dim: 每头 128 维
    4096  // max_seq_len: 最大 4096 tokens
);

std::cout << "Key Cache 大小：" << kv_cache.key_cache.size() << std::endl;
std::cout << "Value Cache 大小：" << kv_cache.value_cache.size() << std::endl;
std::cout << "页表大小：" << kv_cache.page_table.size() << std::endl;
```

**计算大小**:
```
KV Cache 大小 = num_layers × 2 × num_heads × head_dim × max_seq_len
             = 32 × 2 × 32 × 128 × 4096
             = 1,073,741,824 floats ≈ 4GB (FP32)
```

---

### 示例 3: 内存统计监控

```cpp
void print_memory_stats(MemoryManager& manager) {
    MemoryStats stats = manager.get_stats();
    
    std::cout << "=== 内存统计 ===" << std::endl;
    std::cout << "总内存：" << stats.total_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "已分配：" << stats.allocated_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "空闲：" << stats.free_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "利用率：" << stats.utilization() * 100 << "%" << std::endl;
    std::cout << "峰值：" << stats.peak_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "块数量：" << stats.num_blocks << std::endl;
    std::cout << "空闲块：" << stats.free_blocks << std::endl;
}
```

---

## ⚠️ 常见错误

### 错误 1: 未初始化就分配

```cpp
// ❌ 错误
MemoryManager manager(config);
auto* tensor = manager.allocate(1024);  // 失败！

// ✅ 正确
MemoryManager manager(config);
manager.initialize(256 * 1024 * 1024);  // 先初始化
auto* tensor = manager.allocate(1024);
```

---

### 错误 2: 重复释放

```cpp
// ❌ 危险
auto* tensor = manager.allocate(1024);
manager.free(tensor);
manager.free(tensor);  // 重复释放！

// ✅ 正确
auto* tensor = manager.allocate(1024);
manager.free(tensor);
tensor = nullptr;  // 置空防止误用
```

---

### 错误 3: 忽略返回值

```cpp
// ❌ 危险
auto* tensor = manager.allocate(huge_size);
// 直接使用 tensor，可能为 nullptr！

// ✅ 正确
auto* tensor = manager.allocate(huge_size);
if (!tensor) {
    std::cerr << "内存分配失败！" << std::endl;
    return -1;
}
```

---

## 🎯 动手练习

### 练习 1: 实现内存池

尝试实现一个简单的内存池：

```cpp
class SimpleMemoryPool {
public:
    SimpleMemoryPool(size_t size) {
        buffer_.resize(size);
        free_list_.push_back({0, size});
    }
    
    void* allocate(size_t size) {
        // TODO: 实现分配逻辑
        return nullptr;
    }
    
    void free(void* ptr) {
        // TODO: 实现释放逻辑
    }
    
private:
    std::vector<char> buffer_;
    std::vector<std::pair<size_t, size_t>> free_list_;  // {offset, size}
};
```

---

## 📝 总结

✅ **核心概念**:
- 内存池化的原理和优势
- KV Cache 的结构和大小计算
- 分页注意力的内存布局

✅ **实践能力**:
- 使用 MemoryManager 分配/释放内存
- 监控内存使用情况
- 为模型分配 KV Cache

✅ **下一步**:
- 学习 scheduler.cpp 了解任务调度
- 运行 demo_showcase 查看内存管理效果
- 尝试调整 page_size 观察性能变化

---

*最后更新：2024 | LLM-LV1 教学文档系列*
