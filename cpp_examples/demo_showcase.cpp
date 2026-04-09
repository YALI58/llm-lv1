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

/**
 * @brief 打印漂亮的分隔线
 */
void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
}

/**
 * @brief 演示 1: 内存管理器基础
 * 
 * 展示如何配置和使用内存管理器，这是高效 LLM 推理的核心组件
 */
void demo_memory_manager() {
    print_section("演示 1: 智能内存管理器");
    
    // 创建内存配置
    MemoryConfig config;
    config.enable_paged_attention = true;  // 启用分页注意力（vLLM 的核心技术）
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
    
    // 分配 KV Cache（注意力机制的关键数据结构）
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

/**
 * @brief 演示 2: 注意力机制
 * 
 * 展示注意力机制的工作原理，这是 Transformer 模型的核心
 */
void demo_attention() {
    print_section("演示 2: 注意力机制 (Attention Mechanism)");
    
    std::cout << "🧮 注意力机制是 Transformer 的核心组件" << std::endl;
    std::cout << "   公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V\n" << std::endl;
    
    // 设置简单的参数
    int batch_size = 1;
    int num_heads = 2;
    int seq_len = 4;      // 短序列便于演示
    int head_dim = 8;     // 每个头的维度
    
    std::cout << "⚙️  参数设置:" << std::endl;
    std::cout << "   - Batch Size: " << batch_size << std::endl;
    std::cout << "   - 注意力头数：" << num_heads << std::endl;
    std::cout << "   - 序列长度：" << seq_len << std::endl;
    std::cout << "   - 头维度：" << head_dim << std::endl;
    
    // 创建随机的 Q, K, V 矩阵（实际应用中来自模型权重）
    std::vector<float> query(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> key(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> value(batch_size * num_heads * seq_len * head_dim);
    
    // 用简单模式填充数据
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(i % 10) / 10.0f;
        key[i] = static_cast<float>((i + 3) % 10) / 10.0f;
        value[i] = static_cast<float>((i + 7) % 10) / 10.0f;
    }
    
    std::cout << "\n🚀 执行前向传播..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // 调用注意力函数
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
    
    // 显示部分输出
    std::cout << "\n📤 部分输出结果 (前 8 个值):" << std::endl;
    std::cout << "   ";
    for (int i = 0; i < std::min(8, static_cast<int>(output.size())); ++i) {
        std::cout << std::fixed << std::setprecision(4) << output[i];
        if (i < 7) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // 演示带掩码的注意力（用于因果语言建模）
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
        -1.0f,
        &causal_mask
    );
    
    std::cout << "   ✅ 因果掩码注意力计算完成" << std::endl;
    std::cout << "   这确保了每个位置只能看到之前的信息（自回归生成）" << std::endl;
}

/**
 * @brief 演示 3: 推理引擎
 * 
 * 展示如何使用高级推理接口进行文本生成
 */
void demo_inference_engine() {
    print_section("演示 3: 推理引擎 (Inference Engine)");
    
    std::cout << "🤖 推理引擎提供了高级 API 用于文本生成" << std::endl;
    
    // 创建引擎配置
    EngineConfig config;
    config.model_path = "/tmp/demo_model";  // 示例路径
    config.device_type = DeviceType::CPU;   // 使用 CPU
    config.memory.enable_paged_attention = true;
    
    std::cout << "\n⚙️  创建推理引擎..." << std::endl;
    InferenceEngine engine(config);
    
    // 加载模型（这里会创建一个模拟的 tokenizer）
    std::cout << "📥 加载模型..." << std::endl;
    bool loaded = engine.load_model();
    
    if (loaded) {
        std::cout << "✅ 模型加载成功！" << std::endl;
        
        // 查看引擎统计
        auto stats = engine.get_stats();
        std::cout << "\n📊 引擎状态:" << std::endl;
        for (const auto& [key, value] : stats) {
            std::cout << "   - " << key << ": " << value << std::endl;
        }
        
        // 演示文本生成
        std::cout << "\n✍️  文本生成演示..." << std::endl;
        std::string prompt = "Hello, AI!";
        std::cout << "输入提示词: \"" << prompt << "\"" << std::endl;
        
        StoppingCriteria criteria;
        criteria.max_new_tokens = 20;
        criteria.eos_token_id = 2;
        
        std::cout << "\n🚀 生成中..." << std::endl;
        std::string result = engine.generate(prompt, criteria);
        std::cout << "生成结果：" << result << std::endl;
        
        // 演示流式生成
        std::cout << "\n🌊 流式生成演示..." << std::endl;
        std::string stream_prompt = "Once upon";
        std::cout << "输入提示词: \"" << stream_prompt << "\"" << std::endl;
        std::cout << "输出: ";
        
        engine.generate_stream(stream_prompt, [](const std::string& chunk) {
            std::cout << chunk << std::flush;
        }, criteria);
        
        std::cout << "\n\n✅ 流式生成完成!" << std::endl;
        
        // 卸载模型
        engine.unload_model();
        std::cout << "\n📤 模型已卸载" << std::endl;
    } else {
        std::cout << "❌ 模型加载失败" << std::endl;
    }
}

/**
 * @brief 演示 4: 性能特性
 * 
 * 展示框架的性能优化特性
 */
void demo_performance() {
    print_section("演示 4: 性能优化特性");
    
    std::cout << "⚡ LLM-LV1 框架的性能优化亮点:\n" << std::endl;
    
    std::cout << "1️⃣  分页注意力 (Paged Attention)" << std::endl;
    std::cout << "   📝 灵感来自操作系统的虚拟内存分页技术" << std::endl;
    std::cout << "   💡 优势：减少内存碎片，支持更长的上下文" << std::endl;
    std::cout << "   📊 典型提升：2-4 倍吞吐量\n" << std::endl;
    
    std::cout << "2️⃣  内存池化 (Memory Pooling)" << std::endl;
    std::cout << "   📝 预分配内存块，避免频繁的 malloc/free" << std::endl;
    std::cout << "   💡 优势：降低延迟，提高内存利用率" << std::endl;
    std::cout << "   📊 典型提升：30-50% 延迟降低\n" << std::endl;
    
    std::cout << "3️⃣  连续批处理 (Continuous Batching)" << std::endl;
    std::cout << "   📝 动态调整批次大小，充分利用 GPU" << std::endl;
    std::cout << "   💡 优势：提高设备利用率" << std::endl;
    std::cout << "   📊 典型提升：1.5-2 倍吞吐量\n" << std::endl;
    
    std::cout << "4️⃣  KV Cache 优化" << std::endl;
    std::cout << "   📝 缓存注意力键值对，避免重复计算" << std::endl;
    std::cout << "   💡 优势：加速自回归生成" << std::endl;
    std::cout << "   📊 典型提升：5-10 倍生成速度\n" << std::endl;
    
    // 简单的性能演示
    std::cout << "\n🏃 性能测试：批量注意力计算" << std::endl;
    
    int batch_sizes[] = {1, 4, 8, 16};
    int seq_len = 64;
    int num_heads = 8;
    int head_dim = 64;
    
    for (int bs : batch_sizes) {
        std::vector<float> q(bs * num_heads * seq_len * head_dim, 0.5f);
        std::vector<float> k(bs * num_heads * seq_len * head_dim, 0.5f);
        std::vector<float> v(bs * num_heads * seq_len * head_dim, 0.5f);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 运行多次取平均
        int iterations = 10;
        for (int i = 0; i < iterations; ++i) {
            attention_forward(q, k, v, bs, num_heads, seq_len, head_dim, -1.0f, nullptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double avg_time = static_cast<double>(duration.count()) / iterations;
        std::cout << "   Batch=" << std::setw(2) << bs 
                  << " | 平均耗时：" << std::fixed << std::setprecision(2) 
                  << avg_time << " ms" << std::endl;
    }
}

/**
 * @brief 主函数 - 运行所有演示
 */
int main() {
    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                                                          ║" << std::endl;
    std::cout << "║       🚀 欢迎使用 LLM-LV1 深度学习框架 🚀                ║" << std::endl;
    std::cout << "║                                                          ║" << std::endl;
    std::cout << "║   一个从零开始构建的大型语言模型推理引擎                 ║" << std::endl;
    std::cout << "║   Learn LLM Implementation from Scratch!                 ║" << std::endl;
    std::cout << "║                                                          ║" << std::endl;
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
    
    std::cout << "📚 下一步学习建议:" << std::endl;
    std::cout << "   1. 阅读 LEARNING_GUIDE.md 了解详细教程" << std::endl;
    std::cout << "   2. 查看 cpp_tests/ 中的单元测试" << std::endl;
    std::cout << "   3. 尝试修改参数，观察不同配置的效果" << std::endl;
    std::cout << "   4. 添加新的 kernel 或优化现有实现\n" << std::endl;
    
    std::cout << "Happy Coding! 🎊" << std::endl << std::endl;
    
    return 0;
}
