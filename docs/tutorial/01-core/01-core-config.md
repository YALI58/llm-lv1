# config.hpp 完全指南 - 配置系统详解

## 📖 文件概览

**文件位置**: `cpp_include/core/config.hpp`

**作用**: 这是 LLM-LV1 框架的配置中枢，定义了所有可配置的参数，包括：
- 设备类型（CPU、CUDA、Metal 等）
- 量化类型（INT8、INT4、FP8 等）
- 内存管理策略
- 执行策略（线程数、批大小等）

**为什么重要**: 正确的配置是高效推理的前提。这个文件让你能够：
1. 选择运行设备（GPU/CPU）
2. 控制内存使用
3. 调整性能参数
4. 启用/禁用优化特性

---

## 🔍 代码逐行解析

### 1. 头文件保护与包含

```cpp
#ifndef LLM_LV1_CORE_CONFIG_HPP
#define LLM_LV1_CORE_CONFIG_HPP

#include <string>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
```

**解释**:
- `#ifndef/#define/#endif`: 防止头文件被多次包含（C++ 标准做法）
- 包含的库:
  - `<string>`: 字符串处理
  - `<cstdint>`: 固定宽度整数类型（如 `int32_t`）
  - `<memory>`: 智能指针
  - `<unordered_map>`: 哈希表，用于存储键值对
  - `<vector>`: 动态数组

---

### 2. 命名空间声明

```cpp
namespace llm_lv1 {
```

**解释**: 
- 所有代码都在 `llm_lv1` 命名空间内，避免与其他库冲突
- 使用时需要 `llm_lv1::EngineConfig` 或先 `using namespace llm_lv1;`

---

### 3. 设备类型枚举

```cpp
/**
 * @brief Supported device types for inference
 */
enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    WEBGPU,
    METAL
};
```

**详细解释**:

| 设备类型 | 说明 | 适用场景 |
|---------|------|---------|
| `CPU` | 中央处理器 | 调试、小模型、无 GPU 环境 |
| `CUDA` | NVIDIA GPU | 主流选择，性能最佳 |
| `ROCM` | AMD GPU | AMD 显卡用户 |
| `WEBGPU` | 浏览器 GPU | Web 应用 |
| `METAL` | Apple GPU | Mac 用户 |

**为什么用 `enum class` 而不是 `enum`**?
- `enum class` 是强类型枚举，更安全
- 防止不同枚举之间的意外转换
- 必须用 `DeviceType::CUDA` 访问，代码更清晰

---

### 4. 设备类型字符串转换

```cpp
inline std::string device_type_to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::ROCM: return "rocm";
        case DeviceType::WEBGPU: return "webgpu";
        case DeviceType::METAL: return "metal";
        default: return "unknown";
    }
}
```

**作用**: 将枚举转换为字符串，用于：
- 日志输出
- 配置文件读写
- JSON 序列化

**`inline` 关键字**: 建议编译器内联展开，提高性能

---

### 5. 从字符串解析设备类型

```cpp
inline DeviceType device_type_from_string(const std::string& str) {
    if (str == "cpu") return DeviceType::CPU;
    if (str == "cuda") return DeviceType::CUDA;
    if (str == "rocm") return DeviceType::ROCM;
    if (str == "webgpu") return DeviceType::WEBGPU;
    if (str == "metal") return DeviceType::METAL;
    return DeviceType::CPU; // default
}
```

**实际应用场景**:
```cpp
// 从配置文件读取
std::string config_device = "cuda";  // 从 JSON/YAML 读取
DeviceType device = device_type_from_string(config_device);

// 从命令行参数读取
// ./program --device cuda
```

---

### 6. 量化类型枚举

```cpp
enum class QuantizationType {
    NONE,   // 不量化（FP32）
    INT8,   // 8 位整数量化
    INT4,   // 4 位整数量化
    FP8,    // 8 位浮点量化
    AWQ,    // Activation-aware Weight Quantization
    GPTQ    // Generative Pre-trained Transformer Quantization
};
```

**量化知识科普**:

| 量化类型 | 精度 | 内存节省 | 速度提升 | 精度损失 |
|---------|------|---------|---------|---------|
| `NONE` | FP32 | 0% | 0% | 0% |
| `INT8` | 8-bit | 75% | 2-3x | <1% |
| `INT4` | 4-bit | 87.5% | 3-4x | 1-3% |
| `FP8` | 8-bit float | 75% | 2-3x | <2% |

**什么是量化**？
- 将模型权重从高精度（32 位浮点）转换为低精度（8 位整数）
- 大幅减少内存占用和计算量
- 轻微精度损失，通常可接受

---

### 7. 内存配置结构体

```cpp
struct MemoryConfig {
    float max_memory_usage = 0.9f;        // Maximum memory usage ratio
    bool enable_paged_attention = true;   // Enable paged attention
    int page_size = 16;                   // Page size for paged attention
    int max_num_blocks_per_seq = 256;     // Max blocks per sequence
    
    MemoryConfig() = default;
};
```

**参数详解**:

#### `max_memory_usage = 0.9f`
- **含义**: 最大使用 90% 的可用内存
- **为什么不是 1.0**？留出余量防止 OOM（Out Of Memory）
- **调整建议**:
  - GPU 内存充足：设为 0.95
  - 多任务环境：设为 0.7-0.8

#### `enable_paged_attention = true` ⭐
- **含义**: 启用分页注意力机制（vLLM 的核心技术）
- **原理**: 像操作系统虚拟内存一样管理 KV Cache
- **优势**:
  - 减少内存碎片
  - 支持更长上下文
  - 提高吞吐量 2-4 倍

#### `page_size = 16`
- **含义**: 每页存储 16 个 token 的 KV Cache
- **调优**:
  - 短文本为主：减小到 8
  - 长文本为主：增大到 32 或 64

#### `max_num_blocks_per_seq = 256`
- **含义**: 每个序列最多 256 个内存块
- **计算最大长度**: `page_size × max_num_blocks_per_seq = 16 × 256 = 4096 tokens`

---

### 8. 执行配置结构体

```cpp
struct ExecutionConfig {
    int num_threads = -1;                 // -1 means auto-detect
    int batch_size = 1;                   // Default batch size
    int max_sequence_length = 4096;       // Max sequence length
    bool enable_cuda_graph = false;       // Enable CUDA graph optimization
    bool stream_mode = true;              // Enable streaming mode
    
    ExecutionConfig() = default;
};
```

**参数详解**:

#### `num_threads = -1`
- **-1**: 自动检测 CPU 核心数
- **手动设置**: 根据负载调整
  ```cpp
  config.execution.num_threads = 4;  // 固定 4 线程
  ```

#### `batch_size = 1`
- **含义**: 一次处理多少个请求
- **调优**:
  - 交互式应用：保持 1（低延迟）
  - 批量处理：增大到 8、16、32（高吞吐）

#### `max_sequence_length = 4096`
- **含义**: 支持的最大序列长度（prompt + 生成）
- **注意**: 受模型本身限制
  - LLaMA 2: 4096
  - LLaMA 3: 8192
  - 某些模型：128K+

#### `enable_cuda_graph = false`
- **含义**: 启用 CUDA Graph 优化
- **作用**: 预定义 GPU 计算图，减少 kernel 启动开销
- **何时启用**: 固定形状的稳定负载

#### `stream_mode = true`
- **含义**: 启用流式输出
- **效果**: 边生成边返回，用户体验更好

---

### 9. EngineConfig 主配置类

```cpp
class EngineConfig {
public:
    DeviceType device_type = DeviceType::CUDA;
    int device_id = 0;
    std::string model_path;
    QuantizationType quantization = QuantizationType::NONE;
    
    MemoryConfig memory;
    ExecutionConfig execution;
    
    bool debug = false;
    std::string log_level = "info";
    
    // Custom parameters for extensibility
    std::unordered_map<std::string, std::string> extra_params;
    
    EngineConfig() = default;
    
    static EngineConfig from_dict(const std::unordered_map<std::string, std::string>& params);
    std::unordered_map<std::string, std::string> to_dict() const;
    bool validate() const;
};
```

**成员变量详解**:

#### 基础配置
```cpp
DeviceType device_type = DeviceType::CUDA;  // 默认使用 GPU
int device_id = 0;                          // 第 0 号 GPU
std::string model_path;                     // 模型路径（必须设置）
QuantizationType quantization = QuantizationType::NONE;  // 不量化
```

#### 嵌套配置
```cpp
MemoryConfig memory;      // 内存配置（前面定义的 struct）
ExecutionConfig execution; // 执行配置
```

#### 调试选项
```cpp
bool debug = false;           // 调试模式
std::string log_level = "info"; // 日志级别：debug/info/warning/error
```

#### 扩展参数
```cpp
std::unordered_map<std::string, std::string> extra_params;
```
- **用途**: 存储自定义参数，无需修改源码
- **示例**:
  ```cpp
  config.extra_params["custom_kernel"] = "flash_attn_v2";
  config.extra_params["attention_dropout"] = "0.1";
  ```

---

## 💡 实际使用示例

### 示例 1: 基础配置（CPU 推理）

```cpp
#include "core/config.hpp"

using namespace llm_lv1;

int main() {
    EngineConfig config;
    
    // 使用 CPU
    config.device_type = DeviceType::CPU;
    config.model_path = "/path/to/model";
    
    // 调整线程数
    config.execution.num_threads = 8;
    
    // 设置最大长度
    config.execution.max_sequence_length = 2048;
    
    // 启用调试日志
    config.debug = true;
    config.log_level = "debug";
    
    // 现在可以用这个配置创建引擎
    // InferenceEngine engine(config);
    
    return 0;
}
```

---

### 示例 2: GPU 高性能配置

```cpp
EngineConfig config;

// 使用 CUDA GPU
config.device_type = DeviceType::CUDA;
config.device_id = 0;  // 第一块 GPU
config.model_path = "/models/llama-2-7b";

// 启用分页注意力（关键优化！）
config.memory.enable_paged_attention = true;
config.memory.page_size = 32;
config.memory.max_num_blocks_per_seq = 512;
config.memory.max_memory_usage = 0.95f;

// 批处理和流式
config.execution.batch_size = 4;
config.execution.stream_mode = true;
config.execution.max_sequence_length = 4096;

// 不量化（追求最高质量）
config.quantization = QuantizationType::NONE;
```

---

### 示例 3: 低内存量化配置

```cpp
EngineConfig config;

config.device_type = DeviceType::CUDA;
config.model_path = "/models/llama-2-7b-int4";

// 启用 INT4 量化（节省 87.5% 显存）
config.quantization = QuantizationType::INT4;

// 保守的内存配置
config.memory.max_memory_usage = 0.8f;
config.memory.enable_paged_attention = true;
config.memory.page_size = 16;

// 单批次，降低显存峰值
config.execution.batch_size = 1;
config.execution.max_sequence_length = 2048;
```

---

### 示例 4: 从配置文件加载

```cpp
// 假设有一个 JSON 配置文件 config.json:
// {
//     "device_type": "cuda",
//     "model_path": "/models/llama-2-7b",
//     "page_size": "32",
//     "batch_size": "4"
// }

std::unordered_map<std::string, std::string> params;
params["device_type"] = "cuda";
params["model_path"] = "/models/llama-2-7b";
params["page_size"] = "32";
params["batch_size"] = "4";

// 使用 from_dict 静态方法（需要实现）
EngineConfig config = EngineConfig::from_dict(params);
```

---

## ⚠️ 常见错误与解决方案

### 错误 1: 忘记设置 model_path

```cpp
// ❌ 错误
EngineConfig config;
InferenceEngine engine(config);  // 会失败！

// ✅ 正确
EngineConfig config;
config.model_path = "/path/to/model";  // 必须设置！
InferenceEngine engine(config);
```

---

### 错误 2: 内存配置超出限制

```cpp
// ❌ 错误：设置超过 1.0
config.memory.max_memory_usage = 1.5f;  // 无效！

// ✅ 正确
config.memory.max_memory_usage = 0.9f;  // 90%
```

---

### 错误 3: page_size 设置不合理

```cpp
// ❌ 不推荐：太小导致页表过大
config.memory.page_size = 1;  // 每页 1 个 token，效率低

// ❌ 不推荐：太大导致内存浪费
config.memory.page_size = 256;  // 短序列浪费严重

// ✅ 推荐范围
config.memory.page_size = 16;   // 通用
config.memory.page_size = 32;   // 长文本场景
```

---

### 错误 4: 混淆 batch_size 和并发请求

```cpp
// ❌ 误解：认为 batch_size=32 可以同时处理 32 个用户
config.execution.batch_size = 32;  // 延迟会很高！

// ✅ 理解：batch_size 是吞吐量优化，不是并发
// 交互式应用：
config.execution.batch_size = 1;   // 低延迟
config.execution.stream_mode = true;

// 批量离线处理：
config.execution.batch_size = 32;  // 高吞吐
```

---

## 🎯 动手练习

### 练习 1: 创建你的第一个配置

创建一个配置，满足以下要求：
- 使用 CPU 推理
- 4 个线程
- 最大序列长度 1024
- 启用调试日志

<details>
<summary>点击查看答案</summary>

```cpp
EngineConfig config;
config.device_type = DeviceType::CPU;
config.execution.num_threads = 4;
config.execution.max_sequence_length = 1024;
config.debug = true;
config.log_level = "debug";
```
</details>

---

### 练习 2: 优化长文本推理

你要处理平均长度为 8000 tokens 的文档，如何配置？

<details>
<summary>点击查看答案</summary>

```cpp
EngineConfig config;
config.device_type = DeviceType::CUDA;
config.model_path = "/path/to/model";

// 增加页面大小和块数量
config.memory.page_size = 64;
config.memory.max_num_blocks_per_seq = 256;  // 64 * 256 = 16384 tokens

// 允许使用更多内存
config.memory.max_memory_usage = 0.95f;

// 设置最大长度
config.execution.max_sequence_length = 16384;
```
</details>

---

### 练习 3: 多 GPU 配置

假设你有 2 块 GPU，如何配置使用第二块 GPU？

<details>
<summary>点击查看答案</summary>

```cpp
EngineConfig config;
config.device_type = DeviceType::CUDA;
config.device_id = 1;  // 第二块 GPU（从 0 开始计数）
config.model_path = "/path/to/model";
```
</details>

---

## 📚 进阶话题

### 1. 分页注意力深入

分页注意力是本框架的核心优化技术，灵感来自操作系统的虚拟内存：

**传统方式的问题**:
- 连续内存分配导致碎片
- 难以支持可变长度序列
- 内存利用率低

**分页方案**:
- 将 KV Cache 分成固定大小的页
- 使用页表映射逻辑位置→物理页
- 按需分配，用完释放

**伪代码示例**:
```cpp
// 逻辑上：token 0, 1, 2, 3, 4, 5...
// 物理上：页 5, 页 2, 页 5, 页 8, 页 1, 页 3...

page_table[logical_page] = physical_page;
```

---

### 2. 量化技术对比

| 量化方法 | 原理 | 优点 | 缺点 |
|---------|------|------|------|
| **INT8** | 线性映射到 8 位整数 | 成熟稳定 | 需要校准数据 |
| **INT4** | 更激进的量化 | 更高压缩率 | 精度损失较大 |
| **FP8** | 8 位浮点格式 | 硬件支持好 | 较新，兼容性待提升 |
| **AWQ** | 激活感知的权重量化 | 保护重要权重 | 实现复杂 |
| **GPTQ** | 基于 Hessian 的量化 | 精度高 | 量化慢 |

---

### 3. CUDA Graph 优化

**什么是 CUDA Graph**?
- 预先录制 GPU 计算序列
- 一次性提交整个计算图
- 减少 CPU-GPU 通信开销

**何时使用**:
```cpp
// ✅ 适合：固定形状的重复计算
config.execution.enable_cuda_graph = true;

// ❌ 不适合：动态变化的计算
config.execution.enable_cuda_graph = false;
```

---

## 🔗 相关文件

- [engine.hpp](02-core-engine.md) - 学习如何使用配置创建引擎
- [memory_manager.hpp](../03-runtime/01-memory-manager.md) - 深入了解内存管理
- [LEARNING_GUIDE.md](../../../LEARNING_GUIDE.md) - 完整学习路线

---

## 📝 总结

通过本教程，你应该掌握：

✅ **核心概念**:
- 设备类型的选择和转换
- 量化类型及其影响
- 内存配置的关键参数
- 执行策略的调整

✅ **实践能力**:
- 创建适合不同场景的配置
- 诊断和解决配置问题
- 根据硬件资源调整参数

✅ **下一步**:
- 学习 [state.hpp](03-core-state.md) 了解状态管理
- 学习 [engine.hpp](02-core-engine.md) 掌握推理引擎
- 运行 `demo_showcase` 查看实际效果

---

*最后更新：2024 | LLM-LV1 教学文档系列*
