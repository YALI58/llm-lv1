# demo_showcase.cpp 完全解析 - 核心功能演示

## 📖 文件概览

**文件位置**: `cpp_examples/demo_showcase.cpp`

**作用**: 这是 LLM-LV1 框架的**综合演示程序**，展示所有核心功能：
- 内存管理器使用
- 注意力机制计算
- 推理引擎调用
- 性能特性演示

**为什么重要**: 这是学习框架的**最佳起点**，一个文件涵盖所有关键组件。

---

## 🔍 完整代码解析

### 1. 文件头部注释

```cpp
/**
 * @file demo_showcase.cpp
 * @brief 展示 LLM-LV1 项目的核心功能 - 初学者友好演示
 *
 * 这个演示文件展示了如何使用 LLM-LV1 框架构建一个简单的语言模型推理系统。
 * 通过这个例子，你将学习到：
 * 1. 内存管理器的使用（支持分页注意力机制）
 * 2. 注意力机制的实现
 * 3. 推理引擎的基本用法
 * 4. 流式文本生成
 *
 * 编译方法：
 *   mkdir -p build && cd build
 *   cmake ..
 *   make demo_showcase
 *
 * 运行方法：
 *   ./demo_showcase
 */
```

**要点**:
- 清晰的文档是良好代码习惯
- 包含编译和运行说明

---

### 2. 头文件包含

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>

// 包含项目头文件
#include "core/config.hpp"
#include "core/engine.hpp"
#include "runtime/memory_manager.hpp"
#include "kernel/attention.hpp"

using namespace llm_lv1;
```

**标准库说明**:
- `<iostream>`: 输入输出
- `<vector>`: 动态数组
- `<string>`: 字符串
- `<chrono>`: 时间测量（性能测试）
- `<iomanip>`: 格式化输出
- `<cmath>`: 数学函数

**项目头文件**:
- `config.hpp`: 配置系统
- `engine.hpp`: 推理引擎
- `memory_manager.hpp`: 内存管理
- `attention.hpp`: 注意力机制

---

### 3. 辅助函数：打印分隔线

```cpp
void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}
```

**作用**: 美化输出，区分不同演示部分

**输出效果**:
```
============================================================
  演示 1: 智能内存管理器
============================================================
```

---

### 4. 演示 1: 内存管理器

```cpp
void demo_memory_manager() {
    print_section("演示 1: 智能内存管理器");

    // 创建内存配置
    MemoryConfig config;
    config.enable_paged_attention = true;  // 启用分页注意力
    config.page_size = 16;                  // 每页 16 个 token
    config.max_num_blocks_per_seq = 1024;   // 每个序列最多 1024 个块
    config.max_memory_usage = 0.8;          // 使用 80% 的可用内存

    std::cout << "📦 配置内存管理器..." << std::endl;
    std::cout << "   - 分页注意力：" << (config.enable_paged_attention ? "启用" : "禁用") << std::endl;
    std::cout << "   - 页面大小：" << config.page_size << " tokens" << std::endl;
    std::cout << "   - 最大内存使用率：" << (config.max_memory_usage * 100) << "%" << std::endl;

    // 初始化内存管理器
    MemoryManager manager(config);
    size_t total_memory = 256 * 1024 * 1024;  // 256MB
    manager.initialize(total_memory);

    std::cout << "\n✅ 内存管理器初始化成功！" << std::endl;

    // 查看内存统计
    MemoryStats stats = manager.get_stats();
    std::cout << "\n📊 内存统计:" << std::endl;
    std::cout << "   - 总内存：" << stats.total_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "   - 可用内存：" << stats.free_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "   - 内存块数量：" << stats.num_blocks << std::endl;

    // 分配一些内存
    std::cout << "\n🔧 分配内存测试..." << std::endl;
    auto* tensor1 = manager.allocate(1024);  // 分配 1024 个 float
    auto* tensor2 = manager.allocate(2048);  // 分配 2048 个 float

    if (tensor1 && tensor2) {
        std::cout << "   ✅ 成功分配两个张量" << std::endl;
        std::cout << "   - 张量 1 大小：" << tensor1->size() << " floats" << std::endl;
        std::cout << "   - 张量 2 大小：" << tensor2->size() << " floats" << std::endl;

        // 填充一些数据
        for (size_t i = 0; i < tensor1->size(); ++i) {
            (*tensor1)[i] = static_cast<float>(i) / 1000.0f;
        }

        // 释放内存
        manager.free(tensor1);
        manager.free(tensor2);
        std::cout << "   ✅ 内存已释放" << std::endl;
    }

    // 分配 KV Cache
    std::cout << "\n🧠 分配 KV Cache..." << std::endl;
    KVCache kv_cache = manager.allocate_kv_cache(
        2,    // num_layers: 2 层（简化模型）
        4,    // num_heads: 4 个注意力头
        64,   // head_dim: 每个头 64 维
        128   // max_seq_len: 最大序列长度 128
    );

    std::cout << "   ✅ KV Cache 分配成功" << std::endl;
    std::cout << "   - Key Cache 大小：" << kv_cache.key_cache.size() << " floats" << std::endl;
    std::cout << "   - Value Cache 大小：" << kv_cache.value_cache.size() << " floats" << std::endl;
    std::cout << "   - 页面表大小：" << kv_cache.page_table.size() << " entries" << std::endl;

    stats = manager.get_stats();
    std::cout << "\n📈 最终内存利用率：" << (stats.utilization() * 100) << "%" << std::endl;
}
```

**学习要点**:

1. **配置创建**:
   ```cpp
   MemoryConfig config;
   config.enable_paged_attention = true;  // 关键优化！
   ```

2. **初始化和统计**:
   ```cpp
   manager.initialize(256 * 1024 * 1024);  // 256MB
   MemoryStats stats = manager.get_stats();
   ```

3. **内存分配和释放**:
   ```cpp
   auto* tensor = manager.allocate(size);
   manager.free(tensor);
   ```

4. **KV Cache 分配**:
   ```cpp
   KVCache kv_cache = manager.allocate_kv_cache(
       num_layers, num_heads, head_dim, max_seq_len
   );
   ```

---

### 5. 演示 2: 注意力机制

```cpp
void demo_attention() {
    print_section("演示 2: 注意力机制 (Attention Mechanism)");

    std::cout << "🧮 注意力机制是 Transformer 的核心组件" << std::endl;
    std::cout << "   公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V\n" << std::endl;

    // 设置参数
    int batch_size = 1;
    int num_heads = 2;
    int seq_len = 4;
    int head_dim = 8;

    // 创建 Q, K, V
    std::vector<float> query(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> key(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> value(batch_size * num_heads * seq_len * head_dim);

    // 填充数据
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(i % 10) / 10.0f;
        key[i] = static_cast<float>((i + 3) % 10) / 10.0f;
        value[i] = static_cast<float>((i + 7) % 10) / 10.0f;
    }

    // 执行注意力计算
    std::cout << "\n🚀 执行前向传播..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto output = attention_forward(
        query, key, value,
        batch_size, num_heads, seq_len, head_dim,
        -1.0f,  // 自动计算 scale
        nullptr // 无 mask
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "✅ 计算完成!" << std::endl;
    std::cout << "   - 输出大小：" << output.size() << " floats" << std::endl;
    std::cout << "   - 耗时：" << duration.count() << " μs" << std::endl;

    // 演示因果掩码注意力
    std::cout << "\n🎭 演示因果掩码注意力..." << std::endl;
    std::vector<float> causal_mask(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            causal_mask[i * seq_len + j] = -1e9f;  // 遮蔽未来位置
        }
    }

    auto masked_output = attention_forward(
        query, key, value,
        batch_size, num_heads, seq_len, head_dim,
        -1.0f, &causal_mask
    );

    std::cout << "   ✅ 因果掩码注意力计算完成" << std::endl;
}
```

**学习要点**:

1. **张量创建**:
   ```cpp
   std::vector<float> query(batch_size * num_heads * seq_len * head_dim);
   ```

2. **性能测量**:
   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   // ... 计算 ...
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   ```

3. **因果掩码创建**:
   ```cpp
   for (int i = 0; i < seq_len; ++i) {
       for (int j = i + 1; j < seq_len; ++j) {
           causal_mask[i * seq_len + j] = -1e9f;
       }
   }
   ```

---

### 6. 演示 3: 推理引擎

```cpp
void demo_inference_engine() {
    print_section("演示 3: 推理引擎 (Inference Engine)");

    // 创建配置
    EngineConfig config;
    config.model_path = "/tmp/demo_model";
    config.device_type = DeviceType::CPU;
    config.memory.enable_paged_attention = true;

    InferenceEngine engine(config);

    // 加载模型
    bool loaded = engine.load_model();

    if (loaded) {
        // 查看统计
        auto stats = engine.get_stats();

        // 文本生成
        std::string prompt = "Hello, AI!";
        StoppingCriteria criteria;
        criteria.max_new_tokens = 20;
        criteria.eos_token_id = 2;

        std::string result = engine.generate(prompt, criteria);

        // 流式生成
        engine.generate_stream(stream_prompt, 
            [](const std::string& chunk) {
                std::cout << chunk << std::flush;
            }, 
            criteria
        );

        engine.unload_model();
    }
}
```

---

### 7. 演示 4: 性能特性

```cpp
void demo_performance() {
    print_section("演示 4: 性能优化特性");

    std::cout << "⚡ LLM-LV1 框架的性能优化亮点:\n" << std::endl;

    std::cout << "1️⃣  分页注意力 (Paged Attention)" << std::endl;
    std::cout << "   📝 灵感来自操作系统的虚拟内存分页技术" << std::endl;
    std::cout << "   💡 优势：减少内存碎片，支持更长的上下文" << std::endl;
    std::cout << "   📊 典型提升：2-4 倍吞吐量\n" << std::endl;

    // 批量性能测试
    int batch_sizes[] = {1, 4, 8, 16};
    
    for (int bs : batch_sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10; ++i) {
            attention_forward(q, k, v, bs, num_heads, seq_len, head_dim);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double avg_time = static_cast<double>(duration.count()) / 10;
        std::cout << "   Batch=" << bs << " | 平均耗时：" << avg_time << " ms" << std::endl;
    }
}
```

---

### 8. 主函数

```cpp
int main() {
    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       🚀 欢迎使用 LLM-LV1 深度学习框架 🚀                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════╝" << std::endl;

    // 运行所有演示
    demo_memory_manager();
    demo_attention();
    demo_inference_engine();
    demo_performance();

    // 总结
    print_section("🎉 演示完成!");

    std::cout << "通过本演示，你了解了 LLM-LV1 框架的核心功能:\n" << std::endl;
    std::cout << "✅ 智能内存管理 - 支持分页注意力和内存池化" << std::endl;
    std::cout << "✅ 注意力机制 - Transformer 的核心组件" << std::endl;
    std::cout << "✅ 推理引擎 - 统一的文本生成接口" << std::endl;
    std::cout << "✅ 性能优化 - 多种技术提升推理速度\n" << std::endl;

    return 0;
}
```

---

## 🏃 运行示例

```bash
# 编译
cd /workspace
mkdir -p build && cd build
cmake ..
make demo_showcase

# 运行
./demo_showcase
```

**预期输出**:
```
╔══════════════════════════════════════════════════════════╗
║       🚀 欢迎使用 LLM-LV1 深度学习框架 🚀                ║
╚══════════════════════════════════════════════════════════╝

============================================================
  演示 1: 智能内存管理器
============================================================

📦 配置内存管理器...
   - 分页注意力：启用
   - 页面大小：16 tokens
   ...
```

---

## 🎯 动手练习

### 练习 1: 修改参数

尝试修改以下参数并观察效果：
1. `page_size` 从 16 改为 32
2. `batch_size` 从 1 改为 8
3. `seq_len` 从 4 改为 16

### 练习 2: 添加新演示

添加一个新的演示函数，展示 tokenizer 的使用：
```cpp
void demo_tokenizer() {
    print_section("演示：Tokenization");
    
    InferenceEngine engine(config);
    engine.load_model();
    
    std::string text = "Hello, world!";
    auto tokens = engine.tokenize(text);
    
    std::cout << "原文：" << text << std::endl;
    std::cout << "Tokens: ";
    for (int t : tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}
```

---

## 📝 总结

✅ **学到的内容**:
- 内存管理器的配置和使用
- 注意力机制的计算流程
- 推理引擎的基本 API
- 性能测试方法

✅ **下一步**:
- 阅读 simple_inference.cpp 了解更简洁的用法
- 阅读 streaming_example.cpp 学习流式生成
- 基于 demo_showcase 构建自己的应用

---

*最后更新：2024 | LLM-LV1 教学文档系列*
