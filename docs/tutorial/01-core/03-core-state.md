# state.hpp 完全指南 - 状态管理详解

## 📖 文件概览

**文件位置**: `cpp_include/core/state.hpp`

**作用**: 管理文本生成过程中的状态，包括：
- 停止条件判断
- Token 信息记录
- KV Cache 管理
- 生成进度跟踪

---

## 🔍 代码逐行解析

### 1. StoppingCriteria 结构体

```cpp
struct StoppingCriteria {
    int max_new_tokens = 256;       // 最多生成 256 个新 token
    int min_new_tokens = 0;         // 最少生成 0 个新 token
    std::vector<std::string> stop_strings;  // 停止字符串列表
    std::vector<int> stop_token_ids;        // 停止 token ID 列表
    int eos_token_id = -1;          // 结束符 token ID
    float temperature = 1.0f;       // 采样温度
    float top_p = 0.9f;             // Nucleus 采样参数
    int top_k = -1;                 // Top-k 采样（-1 无限制）
    float repetition_penalty = 1.0f; // 重复惩罚
};
```

**参数详解**:

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `max_new_tokens` | 256 | 限制生成长度 | 短回复 50-100，长文 500+ |
| `temperature` | 1.0 | 控制随机性 | <0.7 确定，>1.0 创意 |
| `top_p` | 0.9 | Nucleus 采样 | 0.8-0.95 常用 |
| `repetition_penalty` | 1.0 | 防止重复 | 1.1-1.3 减少重复 |

---

### 2. GenerationState 类

```cpp
class GenerationState {
public:
    std::string request_id;                 // 唯一请求 ID
    std::vector<int> prompt_tokens;         // 输入 prompt tokens
    std::vector<int> generated_tokens;      // 已生成的 tokens
    std::vector<TokenInfo> token_infos;     // 详细 token 信息
    
    std::unordered_map<std::string, void*> attention_cache;  // KV Cache
    
    int current_position = 0;               // 当前位置
    int total_sequence_length = 0;          // 总序列长度
    
    bool is_finished = false;               // 是否完成
    std::string finish_reason;              // 完成原因
};
```

---

## 💡 使用示例

```cpp
// 创建停止条件
StoppingCriteria criteria;
criteria.max_new_tokens = 100;
criteria.temperature = 0.8f;
criteria.top_p = 0.9f;

// 创建生成状态
GenerationState state("request_001");
state.prompt_tokens = engine.tokenize("Hello");

// 逐步生成
while (!state.is_finished) {
    // 生成下一个 token
    int next_token = model.forward(state);
    
    // 添加到状态
    state.append_token(next_token, logprob);
    
    // 检查是否应该停止
    if (criteria.should_stop(state.generated_tokens, "")) {
        state.mark_finished("max_tokens");
    }
}
```

---

## 📝 总结

state.hpp 是推理过程的"记忆系统"，记录所有中间状态。
