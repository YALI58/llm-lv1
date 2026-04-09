# LLM 推理引擎 C++ 版 - 完整指南

## 📚 目录

1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [架构设计](#架构设计)
4. [核心模块详解](#核心模块详解)
5. [API 参考](#api-参考)
6. [使用示例](#使用示例)
7. [学习路径](#学习路径)
8. [常见问题](#常见问题)
9. [性能优化](#性能优化)
10. [扩展开发](#扩展开发)

---

## 项目概述

### 简介

这是一个使用 **C++11/14/17** 标准开发的高性能大语言模型（LLM）推理引擎。项目采用现代化 C++ 特性，提供高效、可扩展的推理解决方案。

### 核心特性

- ✅ **多标准支持**: 兼容 C++11、C++14、C++17
- ✅ **高性能**: 优化的内存管理和并行计算
- ✅ **模块化设计**: 清晰的分层架构，易于扩展
- ✅ **跨平台**: 支持 Linux、macOS、Windows
- ✅ **零依赖**: 最小化外部依赖，便于部署

### 技术栈

| 组件 | 技术选型 |
|------|----------|
| 语言标准 | C++11/14/17 |
| 构建系统 | CMake 3.14+ |
| 并行计算 | OpenMP (可选) |
| GPU 加速 | CUDA (可选) |
| 测试框架 | 自定义轻量级测试 |

---

## 快速开始

### 环境要求

```bash
# 必需
- CMake >= 3.14
- C++ 编译器 (GCC 7+, Clang 5+, MSVC 2017+)
- Make 或 Ninja

# 可选
- OpenMP (多线程加速)
- CUDA Toolkit 11+ (GPU 加速)
```

### 编译步骤

```bash
# 1. 克隆项目
cd /workspace

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 配置项目 (默认 C++17)
cmake ..

# 或使用特定 C++ 标准
cmake .. -DCPP_STANDARD=14  # C++14
cmake .. -DCPP_STANDARD=11  # C++11

# 4. 编译
make -j$(nproc)

# 5. 运行测试
./run_tests

# 6. 运行示例
./simple_inference
./streaming_example
./demo_showcase  # 🌟 强烈推荐！完整的交互式演示
```

### 编译选项

```bash
# 完整配置示例
cmake .. \
    -DCPP_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    -DUSE_OPENMP=ON \
    -DUSE_CUDA=OFF
```

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Application)                  │
│              simple_inference, streaming_example         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   引擎层 (Engine)                        │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│   │   Engine    │  │    State    │  │   Config    │     │
│   └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  运行时层 (Runtime)                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│   │MemoryManager│  │  Scheduler  │  │   Context   │     │
│   └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   内核层 (Kernel)                        │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│   │ Attention   │  │ Activation  │ │Normalization│     │
│   └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   模型层 (Model)                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│   │ModelLoader  │  │ModelRegistry│  │ Tokenizer   │     │
│   └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 目录结构

```
/workspace/
├── cpp_include/           # 头文件
│   ├── core/             # 核心组件
│   │   ├── engine.hpp
│   │   ├── state.hpp
│   │   └── config.hpp
│   ├── kernel/           # 计算内核
│   │   └── attention.hpp
│   └── runtime/          # 运行时
│       └── memory_manager.hpp
├── cpp_src/              # 源文件
│   ├── core/             # 核心实现 (engine.cpp, config.cpp, state.cpp)
│   ├── kernel/           # 内核实现 (attention.cpp, activation.cpp, normalization.cpp)
│   ├── runtime/          # 运行时实现 (memory_manager.cpp, scheduler.cpp, context.cpp)
│   ├── model/            # 模型加载和解析
│   └── utils/            # 工具函数
├── cpp_examples/         # 示例代码
│   ├── demo_showcase.cpp  # 🌟 完整功能演示（推荐初学者）
│   ├── simple_inference.cpp
│   └── streaming_example.cpp
├── cpp_tests/            # 测试代码
│   ├── test_all.cpp      # 综合测试
│   ├── test_engine.cpp   # 引擎测试
│   ├── test_memory_manager.cpp  # 内存管理器测试
│   ├── test_attention.cpp  # 注意力机制测试
│   └── test_scheduler.cpp  # 调度器测试
├── docs/                 # 文档
│   └── tutorial/         # 教程系列
│       ├── INDEX.md      # 教程索引
│       ├── LEARNING_PATH.md  # 学习路径
│       └── [01-06]/      # 分模块教程
├── scripts/              # 脚本工具
│   ├── check_vulnerabilities.sh
│   └── find_new_vulnerabilities.sh
├── CMakeLists.txt        # 构建配置
├── LEARNING_GUIDE.md     # 学习指南
└── README.md             # 本文档
```

---

## 核心模块详解

### 1. 引擎层 (Engine)

**职责**: 推理流程的总控制器

**关键类**:
- `Engine`: 主推理引擎
- `InferenceState`: 推理状态管理
- `EngineConfig`: 引擎配置

**核心流程**:
```cpp
Engine engine(config);
engine.loadModel(model_path);
engine.initialize();

// 推理循环
while (has_input) {
    auto output = engine.generate(input);
    process(output);
}
```

### 2. 运行时层 (Runtime)

**职责**: 资源管理和任务调度

#### MemoryManager
- 内存池管理
- 动态内存分配
- 内存复用优化

#### Scheduler
- 任务调度
- 线程池管理
- 并行执行

#### Context
- 推理上下文
- KV Cache 管理
- 序列状态跟踪

### 3. 内核层 (Kernel)

**职责**: 底层数学运算实现

#### Attention Kernel
```cpp
// 自注意力机制实现
void multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output,
    int num_heads,
    float scale
);
```

#### Activation Functions
- ReLU
- GELU
- SiLU/Swish
- Softmax

#### Normalization
- LayerNorm
- RMSNorm

### 4. 模型层 (Model)

**职责**: 模型加载和解析

- 模型权重加载
- 模型结构解析
- Tokenizer 集成

---

## API 参考

### Engine 类

```cpp
class Engine {
public:
    // 构造函数
    explicit Engine(const EngineConfig& config);
    
    // 模型管理
    bool loadModel(const std::string& model_path);
    void unloadModel();
    
    // 推理接口
    std::vector<int> generate(const std::vector<int>& input_ids,
                              int max_length = 512,
                              float temperature = 1.0f);
    
    // 流式生成
    std::function<void(int)> generateStream(
        const std::vector<int>& input_ids,
        std::function<void(int)> callback);
    
    // 状态管理
    InferenceState getState() const;
    void reset();
};
```

### EngineConfig 类

```cpp
class EngineConfig {
public:
    // 模型配置
    std::string model_path;
    std::string model_type;
    
    // 运行时配置
    int max_seq_length = 2048;
    int batch_size = 1;
    int num_threads = 4;
    
    // 内存配置
    size_t memory_pool_size = 1024 * 1024 * 512;  // 512MB
    
    // 生成配置
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.95f;
    
    // 设备配置
    Device device = Device::CPU;
    bool use_cuda = false;
    bool use_openmp = true;
};
```

### MemoryManager 类

```cpp
class MemoryManager {
public:
    explicit MemoryManager(size_t pool_size);
    
    // 内存分配
    void* allocate(size_t size, size_t alignment = 64);
    void deallocate(void* ptr);
    
    // 内存池统计
    size_t getUsedMemory() const;
    size_t getTotalMemory() const;
    size_t getFreeMemory() const;
    
    // 内存管理
    void clear();
    void defragment();
};
```

---

## 使用示例

### 基础推理示例

```cpp
#include "core/engine.hpp"
#include "core/config.hpp"
#include <iostream>

int main() {
    // 1. 创建配置
    EngineConfig config;
    config.model_path = "models/llama-7b";
    config.max_seq_length = 2048;
    config.num_threads = 8;
    config.temperature = 0.8f;
    
    // 2. 初始化引擎
    Engine engine(config);
    if (!engine.loadModel(config.model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // 3. 准备输入
    std::vector<int> input_ids = {1, 2, 3, 4, 5};  // token IDs
    
    // 4. 生成输出
    auto output = engine.generate(input_ids, 100);
    
    // 5. 处理结果
    for (int token_id : output) {
        std::cout << token_id << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### 流式生成示例

```cpp
#include "core/engine.hpp"
#include <iostream>

int main() {
    EngineConfig config;
    config.model_path = "models/llama-7b";
    
    Engine engine(config);
    engine.loadModel(config.model_path);
    
    std::vector<int> input_ids = {/* prompt tokens */};
    
    // 流式回调
    engine.generateStream(input_ids, [](int new_token) {
        std::cout << decode_token(new_token) << std::flush;
    });
    
    return 0;
}
```

### 批量推理示例

```cpp
#include "core/engine.hpp"
#include <vector>

int main() {
    EngineConfig config;
    config.batch_size = 4;
    config.num_threads = 16;
    
    Engine engine(config);
    engine.loadModel("models/llama-7b");
    
    // 批量输入
    std::vector<std::vector<int>> batch_inputs = {
        {1, 2, 3},
        {4, 5, 6, 7},
        {8, 9},
        {10, 11, 12, 13, 14}
    };
    
    // 批量推理
    auto batch_outputs = engine.batchGenerate(batch_inputs);
    
    return 0;
}
```

---

## 学习路径

### 第一阶段：C++ 基础 (1-2 周)

**目标**: 掌握 C++11/14/17 核心特性

**学习内容**:
1. 智能指针 (`std::unique_ptr`, `std::shared_ptr`)
2. Lambda 表达式
3. 移动语义 (`std::move`, `std::forward`)
4. 模板编程基础
5. constexpr 和编译期计算

**实践任务**:
- 阅读 `cpp_include/core/config.hpp`
- 理解配置类的实现

### 第二阶段：内存管理 (1-2 周)

**目标**: 理解高性能内存管理

**学习内容**:
1. 内存池设计
2. 对齐分配
3. 缓存友好性
4. 内存复用策略

**实践任务**:
- 分析 `cpp_src/runtime/memory_manager.cpp`
- 实现简单的内存池

### 第三阶段：并行计算 (1-2 周)

**目标**: 掌握多线程编程

**学习内容**:
1. OpenMP 基础
2. 线程池实现
3. 任务调度
4. 锁和无锁编程

**实践任务**:
- 研究 `cpp_src/runtime/scheduler.cpp`
- 添加并行矩阵乘法

### 第四阶段：深度学习基础 (2-3 周)

**目标**: 理解 Transformer 架构

**学习内容**:
1. Self-Attention 机制
2. Positional Encoding
3. Layer Normalization
4. Feed-Forward Network

**实践任务**:
- 实现 Attention 核函数
- 添加新的激活函数

### 第五阶段：项目实战 (2-3 周)

**目标**: 完整参与项目开发

**实践任务**:
1. 添加新模型支持
2. 优化现有性能瓶颈
3. 编写单元测试
4. 贡献代码到项目

---

## 常见问题

### Q1: 如何选择 C++ 标准版本？

**A**: 
- **C++11**: 最广泛支持，适合老旧编译器
- **C++14**: 平衡性能和兼容性，推荐选择
- **C++17**: 最新特性，最佳性能，需要较新编译器

```bash
# 检查编译器支持
g++ --version

# C++17 需要 GCC 7+ 或 Clang 5+
```

### Q2: 内存占用过高怎么办？

**A**:
1. 减小 `memory_pool_size` 配置
2. 启用内存复用
3. 使用量化模型
4. 减小 `max_seq_length`

### Q3: 如何启用 GPU 加速？

**A**:
```bash
# 安装 CUDA Toolkit
# 然后编译时启用
cmake .. -DUSE_CUDA=ON
make -j$(nproc)
```

### Q4: 推理速度慢如何优化？

**A**:
1. 增加线程数 (`num_threads`)
2. 启用 OpenMP (`USE_OPENMP=ON`)
3. 使用更小的模型
4. 启用 KV Cache
5. 考虑量化或蒸馏

### Q5: 如何添加自定义算子？

**A**:
1. 在 `cpp_include/kernel/` 创建新头文件
2. 在 `cpp_src/kernel/` 实现源文件
3. 更新 `CMakeLists.txt` 添加源文件
4. 在 Engine 中注册新算子

---

## 性能优化

### 编译优化

```bash
# Release 模式
cmake .. -DCMAKE_BUILD_TYPE=Release

# 添加额外优化标志
export CXXFLAGS="-O3 -march=native -mtune=native"
```

### 运行时优化

```cpp
// 1. 调整线程数
config.num_threads = std::thread::hardware_concurrency();

// 2. 启用内存池
config.memory_pool_size = 1024 * 1024 * 1024;  // 1GB

// 3. 使用合适的批大小
config.batch_size = 8;  // 根据显存调整
```

### 算法优化

1. **KV Cache**: 缓存 Key-Value 避免重复计算
2. **Flash Attention**: 优化 Attention 计算
3. **算子融合**: 合并多个算子减少内存访问
4. **量化**: INT8/FP16 量化加速

---

## 扩展开发

### 添加新模型

```cpp
// 1. 创建模型注册
class NewModelLoader : public ModelLoader {
public:
    bool load(const std::string& path) override {
        // 实现加载逻辑
    }
};

// 2. 注册模型
ModelRegistry::registerModel("new_model", 
    []() { return std::make_unique<NewModelLoader>(); });
```

### 添加新算子

```cpp
// kernel/new_op.hpp
#pragma once
#include "tensor.hpp"

namespace llm_lv1 {
namespace kernel {

void new_operation(const Tensor& input, Tensor& output);

}  // namespace kernel
}  // namespace llm_lv1
```

### 添加新后端

```cpp
// runtime/new_device.cpp
class NewDeviceBackend : public DeviceBackend {
public:
    void* allocate(size_t size) override;
    void copy(void* dst, const void* src, size_t size) override;
    // ...
};
```

---

## 附录

### A. 参考资料

- [C++17 标准文档](https://isocpp.org/std/the-standard)
- [CMake 官方文档](https://cmake.org/documentation/)
- [OpenMP 规范](https://www.openmp.org/specifications/)
- [Transformer 论文](https://arxiv.org/abs/1706.03762)

### B. 社区资源

- GitHub Issues: 报告问题和功能请求
- Discussions: 讨论和交流
- Wiki: 更多详细文档

### C. 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

**最后更新**: 2026 年 04 月 09 日
**维护者**: LLM-LV1 C++ Team
