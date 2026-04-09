# attention.hpp 完全指南 - 注意力机制详解

## 📖 文件概览

**文件位置**: `cpp_include/kernel/attention.hpp`

**作用**: 实现 Transformer 的核心组件——注意力机制，包括：
- 标准 Scaled Dot-Product Attention
- Paged Attention（vLLM 风格优化）
- Flash Attention（高效实现）
- Multi-Head Attention 封装

---

## 🔍 注意力机制基础

### 什么是注意力？

注意力机制让模型能够"关注"输入序列的不同部分。公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**三个关键矩阵**:
- **Q (Query)**: 当前查询向量
- **K (Key)**: 键向量，表示每个位置的特征
- **V (Value)**: 值向量，包含实际信息

**直观理解**:
- Q 和 K 做点积 → 计算相似度
- Softmax → 转换为概率分布
- 用概率加权 V → 得到加权和

---

## 🔍 代码逐行解析

### 1. 标准注意力函数

```cpp
std::vector<float> attention_forward(
    const std::vector<float>& query,      // Q 矩阵
    const std::vector<float>& key,        // K 矩阵
    const std::vector<float>& value,      // V 矩阵
    int batch_size,                       // 批次大小
    int num_heads,                        // 注意力头数
    int seq_len,                          // 序列长度
    int head_dim,                         // 每个头的维度
    float scale = -1.0f,                  // 缩放因子（默认自动计算）
    const std::vector<float>* mask = nullptr  // 可选掩码
);
```

**参数详解**:

| 参数 | 形状 | 说明 |
|------|------|------|
| `query` | [batch, heads, seq_len, head_dim] | Q 矩阵 |
| `key` | [batch, heads, seq_len, head_dim] | K 矩阵 |
| `value` | [batch, heads, seq_len, head_dim] | V 矩阵 |
| `scale` | 标量 | 通常为 1/√head_dim |
| `mask` | [seq_len, seq_len] | 因果掩码或 padding 掩码 |

---

### 2. Paged Attention ⭐

```cpp
std::vector<float> paged_attention_forward(
    const std::vector<float>& query,
    const std::vector<float>& key_cache,   // 分页的 K 缓存
    const std::vector<float>& value_cache, // 分页的 V 缓存
    const std::vector<int>& page_table,    // 页表映射
    const std::vector<int>& context_lengths, // 每个样本的实际长度
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int page_size,        // 每页 token 数
    int num_pages,        // 总页数
    float scale = -1.0f
);
```

**Paged Attention 原理**:

传统方式的问题：
```
连续内存分配 → 碎片化 → 浪费严重
```

Paged Attention 方案：
```
逻辑序列：token 0, 1, 2, 3, 4, 5...
物理存储：页 5, 页 2, 页 5, 页 8, 页 1, 页 3...
                    ↓
              通过页表映射
```

**优势**:
- 减少内存碎片
- 支持更长上下文
- 提高吞吐量 2-4 倍

---

### 3. Multi-Head Attention 类

```cpp
class MultiHeadAttention {
public:
    MultiHeadAttention(int num_heads, int head_dim, float scale = -1.0f);
    
    std::vector<float> forward(
        const std::vector<float>& query,
        const std::vector<float>& key,
        const std::vector<float>& value,
        int batch_size,
        int seq_len,
        const std::vector<float>* mask = nullptr
    );
    
    void update_kv_cache(
        const std::vector<float>& key,
        const std::vector<float>& value,
        int batch_size,
        int seq_len,
        int position
    );
    
private:
    int num_heads_;
    int head_dim_;
    float scale_;
    std::vector<float> kv_cache_;
    bool has_cache_ = false;
};
```

**Multi-Head 的作用**:
- 多个头学习不同的表示子空间
- 类似 CNN 中的多个卷积核
- 增强模型表达能力

---

## 💡 使用示例

### 示例 1: 基础注意力计算

```cpp
#include "kernel/attention.hpp"
#include <iostream>

using namespace llm_lv1;

int main() {
    // 设置参数
    int batch_size = 2;
    int num_heads = 4;
    int seq_len = 8;
    int head_dim = 64;
    
    // 创建随机输入（实际来自模型）
    size_t size = batch_size * num_heads * seq_len * head_dim;
    std::vector<float> Q(size, 0.5f);
    std::vector<float> K(size, 0.5f);
    std::vector<float> V(size, 0.5f);
    
    // 执行注意力
    auto output = attention_forward(Q, K, V, 
                                    batch_size, num_heads, seq_len, head_dim);
    
    std::cout << "输出大小：" << output.size() << std::endl;
    // 输出：[2, 4, 8, 64] = 4096
    
    return 0;
}
```

---

### 示例 2: 因果掩码注意力

```cpp
// 创建因果掩码（防止看到未来）
std::vector<float> causal_mask(seq_len * seq_len, 0.0f);
for (int i = 0; i < seq_len; ++i) {
    for (int j = i + 1; j < seq_len; ++j) {
        causal_mask[i * seq_len + j] = -1e9f;  // 遮蔽未来位置
    }
}

// 使用掩码
auto output = attention_forward(Q, K, V, 
                                batch_size, num_heads, seq_len, head_dim,
                                -1.0f, &causal_mask);
```

**掩码效果**:
```
无掩码: 所有位置互相可见
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]

因果掩码: 只能看到过去和现在
[[0, -∞, -∞],
 [0,   0, -∞],
 [0,   0,   0]]
```

---

### 示例 3: Multi-Head Attention

```cpp
MultiHeadAttention mha(num_heads, head_dim);

// 前向传播
auto output = mha.forward(Q, K, V, batch_size, seq_len);

// 更新 KV Cache（用于自回归生成）
mha.update_kv_cache(K, V, batch_size, seq_len, position);
```

---

## ⚠️ 常见错误

### 错误 1: 忘记除以 sqrt(head_dim)

```cpp
// ❌ 错误：没有缩放
score = Q @ K;  // 数值可能很大

// ✅ 正确
scale = 1.0 / sqrt(head_dim);
score = (Q @ K) * scale;
```

**为什么需要缩放**？
- 防止点积结果过大
- 避免 softmax 梯度消失

---

### 错误 2: 掩码应用时机错误

```cpp
// ❌ 错误：softmax 后应用掩码
scores = softmax(Q @ K);
scores += mask;  // 无效！

// ✅ 正确：softmax 前应用掩码
scores = Q @ K;
scores += mask;  // 遮蔽大负数
scores = softmax(scores);
```

---

### 错误 3: 张量形状不匹配

```cpp
// ❌ 错误：形状不对
// Q: [batch, seq_len, heads, head_dim]  // 错的！

// ✅ 正确
// Q: [batch, heads, seq_len, head_dim]
```

---

## 🎯 动手练习

### 练习 1: 实现简单的注意力

给定以下输入：
```cpp
std::vector<float> Q = {1.0, 0.0, 0.0};
std::vector<float> K = {1.0, 0.0, 0.0};
std::vector<float> V = {0.5, 0.3, 0.2};
```

手动计算 Q·K 并应用 softmax。

<details>
<summary>点击查看答案</summary>

```
Q·K = 1.0*1.0 + 0.0*0.0 + 0.0*0.0 = 1.0
softmax([1.0]) = [1.0]
output = 1.0 * V = [0.5, 0.3, 0.2]
```
</details>

---

## 📝 总结

✅ **核心概念**:
- Scaled Dot-Product Attention 公式
- Multi-Head 机制
- Paged Attention 优化

✅ **实践能力**:
- 调用 attention_forward
- 创建和使用掩码
- 理解 KV Cache

✅ **下一步**:
- 学习 activation.cpp 了解激活函数
- 学习 normalization.cpp 了解归一化
- 运行 demo_showcase 查看效果

---

*最后更新：2024 | LLM-LV1 教学文档系列*
