# engine.hpp 完全指南 - 推理引擎详解

## 📖 文件概览

**文件位置**: `cpp_include/core/engine.hpp`

**作用**: 这是 LLM-LV1 框架的核心推理引擎，提供：
- 模型加载和管理
- 文本生成（同步/流式）
- Tokenization/Detokenization
- 多请求并发处理

**为什么重要**: 这是你与框架交互的主要接口，所有推理任务都通过它完成。

---

## 🔍 代码逐行解析

### 1. 头文件包含与前向声明

```cpp
#ifndef LLM_LV1_CORE_ENGINE_HPP
#define LLM_LV1_CORE_ENGINE_HPP

#include "core/config.hpp"
#include "core/state.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace llm_lv1 {

// Forward declarations
class MemoryManager;
class Tokenizer;
```

**解释**:
- 依赖 `config.hpp`（配置）和 `state.hpp`（状态管理）
- **前向声明**: 提前声明 `MemoryManager` 和 `Tokenizer`，避免循环依赖
- `<functional>`: 用于回调函数（流式生成）

---

### 2. 类文档注释

```cpp
/**
 * @brief Main inference engine for LLM text generation
 * 
 * This engine provides a unified interface for loading models and
 * generating text with various sampling strategies. It supports
 * both synchronous and asynchronous generation, as well as
 * streaming output.
 * 
 * Example usage:
 * @code
 * EngineConfig config;
 * config.device_type = DeviceType::CUDA;
 * config.model_path = "/path/to/model";
 * 
 * InferenceEngine engine(config);
 * 
 * if (engine.load_model()) {
 *     std::string result = engine.generate("Hello, how are you?");
 *     std::cout << result << std::endl;
 *     
 *     // Streaming generation
 *     engine.generate_stream("Tell me a story", [](const std::string& chunk) {
 *         std::cout << chunk << std::flush;
 *     });
 * }
 * @endcode
 */
```

**要点**:
- 清晰的 API 文档是使用库的关键
- 示例代码展示了最基本的使用流程

---

### 3. 构造函数与析构函数

```cpp
class InferenceEngine {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit InferenceEngine(const EngineConfig& config = EngineConfig());
    
    /**
     * @brief Destructor
     */
    ~InferenceEngine();
    
    // Disable copy
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
```

**解释**:

#### `explicit` 关键字
- 防止隐式类型转换
- 必须显式调用构造函数

```cpp
// ❌ 错误（如果有 implicit）
InferenceEngine engine = some_config;  // 可能意外转换

// ✅ 正确
InferenceEngine engine(some_config);
```

#### 默认参数
```cpp
InferenceEngine(const EngineConfig& config = EngineConfig());
```
- 可以不带参数创建：`InferenceEngine engine;`
- 使用默认配置（CPU，无模型路径）

#### 禁用拷贝
```cpp
InferenceEngine(const InferenceEngine&) = delete;
```
- **原因**: 引擎管理大量资源（内存、GPU 上下文）
- 拷贝会导致资源重复释放
- 如需传递，使用引用或指针

---

### 4. 模型加载接口

```cpp
    /**
     * @brief Load the model and tokenizer
     * @param model_path Path to model files (optional, uses config if not provided)
     * @param tokenizer_path Path to tokenizer files (optional, uses model_path if not provided)
     * @return true if loading successful
     */
    bool load_model(const std::string& model_path = "", const std::string& tokenizer_path = "");
    
    /**
     * @brief Unload the model and free resources
     */
    void unload_model();
    
    /**
     * @brief Check if model is loaded
     */
    bool is_loaded() const { return is_loaded_; }
```

**使用示例**:

```cpp
// 方式 1: 使用配置中的路径
EngineConfig config;
config.model_path = "/models/llama-2-7b";
InferenceEngine engine(config);
engine.load_model();  // 使用 config.model_path

// 方式 2: 覆盖配置路径
engine.load_model("/models/other-model");

// 方式 3: 分别指定模型和 tokenizer
engine.load_model("/models/llama", "/models/llama-tokenizer");

// 检查加载状态
if (engine.is_loaded()) {
    std::cout << "模型已加载" << std::endl;
}

// 卸载模型（释放显存）
engine.unload_model();
```

---

### 5. 同步文本生成

```cpp
    /**
     * @brief Generate text synchronously
     * @param prompt Input prompt text
     * @param stopping_criteria Criteria for stopping generation
     * @return Generated text string
     */
    std::string generate(
        const std::string& prompt,
        const StoppingCriteria& stopping_criteria = StoppingCriteria()
    );
```

**完整示例**:

```cpp
InferenceEngine engine(config);
engine.load_model();

// 简单生成
std::string result = engine.generate("Hello, AI!");
std::cout << result << std::endl;

// 带停止条件
StoppingCriteria criteria;
criteria.max_new_tokens = 100;      // 最多 100 个新 token
criteria.temperature = 0.8f;        // 降低随机性
criteria.top_p = 0.9f;              // Nucleus sampling
criteria.stop_strings = {"END"};    // 遇到"END"停止

std::string controlled = engine.generate("Write a story", criteria);
```

---

### 6. 流式文本生成 ⭐

```cpp
    /**
     * @brief Generate text with streaming output
     * @param prompt Input prompt text
     * @param callback Callback function called for each generated chunk
     * @param stopping_criteria Criteria for stopping generation
     */
    void generate_stream(
        const std::string& prompt,
        const std::function<void(const std::string&)>& callback,
        const StoppingCriteria& stopping_criteria = StoppingCriteria()
    );
```

**为什么需要流式**？
- 用户无需等待完整响应
- 更好的用户体验（类似 ChatGPT）
- 可以提前中断生成

**使用示例**:

```cpp
// Lambda 回调
engine.generate_stream("讲一个故事", [](const std::string& chunk) {
    std::cout << chunk << std::flush;  // 立即输出
});

// 带停止条件的流式
StoppingCriteria criteria;
criteria.max_new_tokens = 50;

engine.generate_stream("写一首诗", 
    [](const std::string& chunk) {
        std::cout << chunk << std::flush;
    },
    criteria
);

// 收集到字符串
std::string full_response;
engine.generate_stream("Hello", 
    [&full_response](const std::string& chunk) {
        full_response += chunk;  // 累积
    }
);
std::cout << "\n完整回复：" << full_response << std::endl;
```

---

### 7. Tokenization 接口

```cpp
    /**
     * @brief Tokenize input text
     * @param text Input text to tokenize
     * @return List of token IDs
     */
    std::vector<int> tokenize(const std::string& text);
    
    /**
     * @brief Convert token IDs back to text
     * @param tokens List of token IDs
     * @return Decoded text string
     */
    std::string detokenize(const std::vector<int>& tokens);
```

**什么是 Tokenization**?
- 将文本切分成模型能理解的单元（tokens）
- 例如："Hello world" → [15496, 2345]

**使用示例**:

```cpp
// 分词
std::vector<int> tokens = engine.tokenize("Hello, world!");
std::cout << "Token 数量：" << tokens.size() << std::endl;
for (int id : tokens) {
    std::cout << id << " ";
}

// 还原
std::string text = engine.detokenize(tokens);
std::cout << "还原后：" << text << std::endl;

// 实际应用：计算 prompt 长度
std::string prompt = "这是一个很长的输入...";
auto tokens = engine.tokenize(prompt);
if (tokens.size() > 4096) {
    std::cout << "警告：prompt 超出最大长度！" << std::endl;
}
```

---

### 8. 统计信息接口

```cpp
    /**
     * @brief Get engine statistics
     */
    std::unordered_map<std::string, std::string> get_stats() const;
```

**返回内容示例**:

```cpp
auto stats = engine.get_stats();
for (const auto& [key, value] : stats) {
    std::cout << key << ": " << value << std::endl;
}

// 可能输出：
// model_loaded: true
// model_path: /models/llama-2-7b
// device: cuda
// memory_used: 14523MB
// total_requests: 42
// tokens_generated: 15234
```

---

## 💡 完整使用示例

### 示例 1: 最简单的对话机器人

```cpp
#include "core/engine.hpp"
#include <iostream>

using namespace llm_lv1;

int main() {
    // 1. 配置
    EngineConfig config;
    config.model_path = "/models/llama-2-7b-chat";
    config.device_type = DeviceType::CUDA;
    
    // 2. 创建引擎
    InferenceEngine engine(config);
    
    // 3. 加载模型
    if (!engine.load_model()) {
        std::cerr << "模型加载失败！" << std::endl;
        return 1;
    }
    
    std::cout << "🤖 聊天机器人已启动（输入 'quit' 退出）\n" << std::endl;
    
    // 4. 对话循环
    while (true) {
        std::cout << "你：";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "quit") break;
        
        // 5. 生成回复
        std::string response = engine.generate(user_input);
        std::cout << "AI: " << response << std::endl;
    }
    
    // 6. 清理
    engine.unload_model();
    return 0;
}
```

---

### 示例 2: 流式聊天机器人

```cpp
#include "core/engine.hpp"
#include <iostream>

using namespace llm_lv1;

int main() {
    EngineConfig config;
    config.model_path = "/models/llama-2-7b-chat";
    config.device_type = DeviceType::CUDA;
    config.execution.stream_mode = true;
    
    InferenceEngine engine(config);
    engine.load_model();
    
    std::cout << "🤖 流式聊天机器人（输入 'quit' 退出）\n" << std::endl;
    
    while (true) {
        std::cout << "你：";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        if (user_input == "quit") break;
        
        std::cout << "AI: ";
        
        // 流式输出
        engine.generate_stream(user_input, 
            [](const std::string& chunk) {
                std::cout << chunk << std::flush;
            }
        );
        
        std::cout << "\n\n";
    }
    
    engine.unload_model();
    return 0;
}
```

---

### 示例 3: 批量文本生成

```cpp
#include "core/engine.hpp"
#include <vector>
#include <fstream>

using namespace llm_lv1;

int main() {
    EngineConfig config;
    config.model_path = "/models/llama-2-7b";
    config.device_type = DeviceType::CUDA;
    config.execution.batch_size = 4;  // 批处理
    
    InferenceEngine engine(config);
    engine.load_model();
    
    // 读取 prompts
    std::vector<std::string> prompts = {
        "写一篇关于人工智能的短文",
        "解释量子计算的原理",
        "创作一首关于春天的诗",
        "描述未来的城市生活"
    };
    
    // 批量生成
    std::ofstream output("results.txt");
    
    StoppingCriteria criteria;
    criteria.max_new_tokens = 200;
    
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::cout << "生成 [" << (i+1) << "/" << prompts.size() << "]: " 
                  << prompts[i] << std::endl;
        
        std::string result = engine.generate(prompts[i], criteria);
        
        output << "=== Prompt " << (i+1) << " ===" << std::endl;
        output << prompts[i] << std::endl;
        output << "\n=== Response ===" << std::endl;
        output << result << std::endl;
        output << std::endl;
    }
    
    std::cout << "完成！结果已保存到 results.txt" << std::endl;
    
    engine.unload_model();
    return 0;
}
```

---

### 示例 4: 带超时的生成

```cpp
#include "core/engine.hpp"
#include <future>
#include <chrono>

using namespace llm_lv1;

std::string generate_with_timeout(
    InferenceEngine& engine,
    const std::string& prompt,
    int timeout_seconds = 30
) {
    auto future = std::async(std::launch::async, 
        [&engine, prompt]() {
            return engine.generate(prompt);
        }
    );
    
    auto status = future.wait_for(std::chrono::seconds(timeout_seconds));
    
    if (status == std::future_status::ready) {
        return future.get();
    } else {
        throw std::runtime_error("生成超时！");
    }
}

int main() {
    EngineConfig config;
    config.model_path = "/models/llama-2-7b";
    InferenceEngine engine(config);
    engine.load_model();
    
    try {
        std::string result = generate_with_timeout(engine, "写一个长故事", 10);
        std::cout << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
    }
    
    engine.unload_model();
    return 0;
}
```

---

## ⚠️ 常见错误与解决方案

### 错误 1: 未加载模型就生成

```cpp
// ❌ 错误
InferenceEngine engine(config);
engine.generate("Hello");  // 崩溃！

// ✅ 正确
InferenceEngine engine(config);
if (engine.load_model()) {
    engine.generate("Hello");
}
```

---

### 错误 2: 忘记检查返回值

```cpp
// ❌ 危险
engine.load_model();  // 可能失败
engine.generate("...");

// ✅ 安全
if (!engine.load_model()) {
    std::cerr << "加载失败" << std::endl;
    return 1;
}
```

---

### 错误 3: 流式回调中做耗时操作

```cpp
// ❌ 不好：回调中睡眠
engine.generate_stream("Hello", [](const std::string& chunk) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 阻塞！
    std::cout << chunk << std::flush;
});

// ✅ 好：回调只做快速输出
engine.generate_stream("Hello", [](const std::string& chunk) {
    std::cout << chunk << std::flush;  // 立即返回
});
```

---

### 错误 4: 多次拷贝引擎

```cpp
// ❌ 错误：编译失败（拷贝被禁用）
InferenceEngine engine1(config);
InferenceEngine engine2 = engine1;  // 错误！

// ✅ 正确：使用引用
void process(InferenceEngine& engine) {
    engine.generate("Hello");
}

process(engine1);
```

---

## 🎯 动手练习

### 练习 1: 创建问答机器人

编写一个程序，循环读取用户问题并生成答案，直到用户输入"exit"。

<details>
<summary>参考代码</summary>

```cpp
InferenceEngine engine(config);
engine.load_model();

while (true) {
    std::cout << "Q: ";
    std::string question;
    std::getline(std::cin, question);
    
    if (question == "exit") break;
    
    std::string answer = engine.generate(question);
    std::cout << "A: " << answer << std::endl;
}
```
</details>

---

### 练习 2: 实现简单的文本摘要

给定一篇长文章，使用引擎生成摘要（限制在 50 个 token 内）。

<details>
<summary>提示</summary>

```cpp
StoppingCriteria criteria;
criteria.max_new_tokens = 50;

std::string article = "这是一篇很长的文章...";
std::string prompt = "请总结以下文章：" + article;
std::string summary = engine.generate(prompt, criteria);
```
</details>

---

## 📝 总结

通过本教程，你应该掌握：

✅ **核心 API**:
- `load_model()` / `unload_model()`
- `generate()` / `generate_stream()`
- `tokenize()` / `detokenize()`
- `get_stats()`

✅ **实践能力**:
- 创建基本的对话应用
- 实现流式输出
- 处理错误情况

✅ **下一步**:
- 学习 [state.hpp](03-core-state.md) 深入理解生成状态
- 查看 [demo_showcase.cpp](../../../cpp_examples/demo_showcase.cpp) 完整示例
- 尝试修改参数观察效果

---

*最后更新：2024 | LLM-LV1 教学文档系列*
