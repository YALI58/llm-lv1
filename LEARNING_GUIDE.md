# LLM 推理引擎 C++ 版 - 学习教程

## 🎓 完整学习路线

本教程将带你从零开始，系统学习如何使用 C++11/14/17 开发高性能 LLM 推理引擎。

---

## 第一部分：C++ 现代特性速成

### 1.1 智能指针 (C++11)

**为什么需要智能指针？**
- 自动管理内存，避免内存泄漏
- 清晰的资源所有权语义

```cpp
#include <memory>

// unique_ptr - 独占所有权
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
// 不需要手动 delete，离开作用域自动释放

// shared_ptr - 共享所有权
std::shared_ptr<int> ptr2 = std::make_shared<int>(42);
std::shared_ptr<int> ptr3 = ptr2;  // 引用计数 +1
// 最后一个 shared_ptr 销毁时释放内存

// weak_ptr - 弱引用，不增加引用计数
std::weak_ptr<int> weak = ptr2;
if (auto locked = weak.lock()) {
    // 安全使用
}
```

**实践练习**:
```cpp
// 在 config.hpp 中查找智能指针的使用
// 思考：为什么这里使用 unique_ptr 而不是 shared_ptr？
```

### 1.2 Lambda 表达式 (C++11)

```cpp
#include <algorithm>
#include <vector>

std::vector<int> nums = {1, 2, 3, 4, 5};

// 基本用法
std::for_each(nums.begin(), nums.end(), 
    [](int n) { 
        std::cout << n << " "; 
    });

// 捕获外部变量
int factor = 2;
std::transform(nums.begin(), nums.end(), nums.begin(),
    [factor](int n) { return n * factor; });

// 通用 Lambda (C++14)
auto generic_lambda = [](auto x, auto y) { 
    return x + y; 
};
```

**在推理引擎中的应用**:
```cpp
// 流式生成回调
engine.generateStream(input_ids, [](int new_token) {
    std::cout << decode(new_token) << std::flush;
});
```

### 1.3 移动语义 (C++11)

```cpp
#include <vector>
#include <utility>

class Tensor {
public:
    std::vector<float> data;
    
    // 拷贝构造函数
    Tensor(const Tensor& other) : data(other.data) {}
    
    // 移动构造函数 (高效!)
    Tensor(Tensor&& other) noexcept 
        : data(std::move(other.data)) {}
    
    // 移动赋值运算符
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }
};

// 使用 std::move 触发移动
Tensor t1;
Tensor t2 = std::move(t1);  // 不复制数据，只转移所有权
```

**性能对比**:
```cpp
// 慢：拷贝
std::vector<Tensor> tensors;
tensors.push_back(large_tensor);  // 调用拷贝构造

// 快：移动
tensors.push_back(std::move(large_tensor));  // 调用移动构造
```

### 1.4 constexpr 和编译期计算 (C++11/14/17)

```cpp
// C++11: 简单的 constexpr
constexpr int square(int x) {
    return x * x;
}
constexpr int result = square(5);  // 在编译期计算

// C++14: 更复杂的逻辑
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// C++17: if constexpr
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // 整数处理
    } else if constexpr (std::is_floating_point_v<T>) {
        // 浮点数处理
    }
}
```

---

## 第二部分：内存管理深入

### 2.1 内存池设计原理

**问题**: 频繁的 new/delete 会导致：
1. 内存碎片
2. 分配开销大
3. 缓存不友好

**解决方案**: 内存池

```cpp
class MemoryPool {
private:
    std::vector<char> pool_;      // 预分配的内存块
    std::vector<size_t> free_list_;  // 空闲块列表
    size_t block_size_;
    
public:
    MemoryPool(size_t block_size, size_t num_blocks)
        : block_size_(block_size) {
        // 预分配内存
        pool_.resize(block_size * num_blocks);
        
        // 初始化空闲列表
        for (size_t i = 0; i < num_blocks; ++i) {
            free_list_.push_back(i);
        }
    }
    
    void* allocate() {
        if (free_list_.empty()) {
            return nullptr;  // 内存池耗尽
        }
        size_t index = free_list_.back();
        free_list_.pop_back();
        return pool_.data() + index * block_size_;
    }
    
    void deallocate(void* ptr) {
        size_t offset = static_cast<char*>(ptr) - pool_.data();
        size_t index = offset / block_size_;
        free_list_.push_back(index);
    }
};
```

**实践任务**:
```cpp
// 分析 cpp_src/runtime/memory_manager.cpp
// 1. 找出内存池的实现
// 2. 理解对齐分配的原理
// 3. 尝试添加内存统计功能
```

### 2.2 内存对齐

```cpp
#include <cstddef>

// 为什么需要对齐？
// CPU 访问对齐的内存更快

// C++11: alignof 和 alignas
struct AlignedStruct {
    alignas(64) char data[100];  // 64 字节对齐
};

static_assert(alignof(AlignedStruct) == 64, "Must be 64-byte aligned");

// 对齐分配
void* aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#if __cplusplus >= 201703L
    ptr = ::aligned_alloc(alignment, size);
#else
    posix_memalign(&ptr, alignment, size);
#endif
    return ptr;
}
```

### 2.3 缓存友好性

```cpp
// 不好的设计：随机访问
struct BadLayout {
    int* data1;
    int* data2;
    int* data3;
    // 每次访问可能 cache miss
};

// 好的设计：连续存储
struct GoodLayout {
    std::vector<int> data1;
    std::vector<int> data2;
    std::vector<int> data3;
    // 顺序访问，cache friendly
};

// AoS vs SoA
struct Particle_AoS {
    float x, y, z;
    float vx, vy, vz;
};

// SoA (Structure of Arrays) - 更适合 SIMD
struct Particle_SoA {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
};
```

---

## 第三部分：并行计算实战

### 3.1 OpenMP 基础

```cpp
#include <omp.h>
#include <vector>

// 并行 for 循环
std::vector<float> compute(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * 2.0f + 1.0f;
    }
    
    return output;
}

// 并行归约
float sum(const std::vector<float>& vec) {
    float result = 0.0f;
    
    #pragma omp parallel for reduction(+:result)
    for (float v : vec) {
        result += v;
    }
    
    return result;
}

// 嵌套并行
#pragma omp parallel num_threads(4)
{
    #pragma omp single
    {
        #pragma omp task
        task1();
        
        #pragma omp task
        task2();
    }
}
```

### 3.2 线程池实现

```cpp
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
    
public:
    ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }
};

// 使用示例
ThreadPool pool(4);
for (int i = 0; i < 100; ++i) {
    pool.enqueue([i] {
        std::cout << "Task " << i << std::endl;
    });
}
```

### 3.3 矩阵乘法并行化

```cpp
// 串行版本
void matmul_serial(const float* A, const float* B, float* C, 
                   int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// OpenMP 并行版本
void matmul_parallel(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 分块优化版本 (更好的缓存利用)
void matmul_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, int tile_size = 32) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += tile_size) {
        for (int j = 0; j < N; j += tile_size) {
            for (int k = 0; k < K; k += tile_size) {
                for (int ii = i; ii < std::min(i + tile_size, M); ++ii) {
                    for (int jj = j; jj < std::min(j + tile_size, N); ++jj) {
                        float sum = C[ii * N + jj];
                        for (int kk = k; kk < std::min(k + tile_size, K); ++kk) {
                            sum += A[ii * K + kk] * B[kk * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            }
        }
    }
}
```

---

## 第四部分：Transformer 架构解析

### 4.1 Self-Attention 机制

**数学公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**代码实现**:
```cpp
void self_attention(
    const float* Q,  // [seq_len, d_model]
    const float* K,  // [seq_len, d_model]
    const float* V,  // [seq_len, d_model]
    float* output,   // [seq_len, d_model]
    int seq_len,
    int d_model,
    int d_head
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    
    // 1. 计算 QK^T
    std::vector<float> scores(seq_len * seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                dot += Q[i * d_model + d] * K[j * d_model + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    
    // 2. Softmax
    for (int i = 0; i < seq_len; ++i) {
        float max_val = scores[i * seq_len];
        for (int j = 1; j < seq_len; ++j) {
            max_val = std::max(max_val, scores[i * seq_len + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            float exp_val = std::exp(scores[i * seq_len + j] - max_val);
            scores[i * seq_len + j] = exp_val;
            sum += exp_val;
        }
        
        for (int j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] /= sum;
        }
    }
    
    // 3. 乘以 V
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < d_model; ++d) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                val += scores[i * seq_len + j] * V[j * d_model + d];
            }
            output[i * d_model + d] = val;
        }
    }
}
```

### 4.2 Multi-Head Attention

```cpp
void multi_head_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len,
    int d_model,
    int num_heads
) {
    int d_head = d_model / num_heads;
    
    std::vector<float> head_output(d_model * seq_len);
    
    // 对每个 head 并行计算
    #pragma omp parallel for
    for (int h = 0; h < num_heads; ++h) {
        int head_offset = h * d_head;
        
        // 提取当前 head 的 Q, K, V
        std::vector<float> Q_head(seq_len * d_head);
        std::vector<float> K_head(seq_len * d_head);
        std::vector<float> V_head(seq_len * d_head);
        
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < d_head; ++d) {
                Q_head[i * d_head + d] = Q[i * d_model + head_offset + d];
                K_head[i * d_head + d] = K[i * d_model + head_offset + d];
                V_head[i * d_head + d] = V[i * d_model + head_offset + d];
            }
        }
        
        // 计算 attention
        std::vector<float> head_result(seq_len * d_head);
        self_attention(Q_head.data(), K_head.data(), V_head.data(),
                      head_result.data(), seq_len, d_head, d_head);
        
        // 合并结果
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < d_head; ++d) {
                head_output[i * d_model + head_offset + d] = 
                    head_result[i * d_head + d];
            }
        }
    }
    
    // 复制到输出
    std::copy(head_output.begin(), head_output.end(), output);
}
```

### 4.3 Layer Normalization

```cpp
void layer_norm(
    const float* input,
    float* output,
    int hidden_size,
    float eps = 1e-5f
) {
    // 1. 计算均值
    float mean = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        mean += input[i];
    }
    mean /= hidden_size;
    
    // 2. 计算方差
    float variance = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= hidden_size;
    
    // 3. 归一化
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (int i = 0; i < hidden_size; ++i) {
        output[i] = (input[i] - mean) * inv_std;
    }
}

// RMSNorm (更高效的变体)
void rms_norm(
    const float* input,
    float* output,
    int hidden_size,
    float eps = 1e-5f
) {
    // 计算均方根
    float rms = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        rms += input[i] * input[i];
    }
    rms = std::sqrt(rms / hidden_size + eps);
    
    // 归一化
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < hidden_size; ++i) {
        output[i] = input[i] * inv_rms;
    }
}
```

### 4.4 Positional Encoding

```cpp
void positional_encoding(
    float* embedding,
    int seq_len,
    int d_model
) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d_model; i += 2) {
            float div_term = std::exp(-2.0f * std::log(10000.0f) * 
                                      (i / 2) / d_model);
            
            // sin for even indices
            embedding[pos * d_model + i] = 
                std::sin(pos * div_term);
            
            // cos for odd indices
            if (i + 1 < d_model) {
                embedding[pos * d_model + i + 1] = 
                    std::cos(pos * div_term);
            }
        }
    }
}
```

---

## 第五部分：项目实战演练

### 5.1 阅读源码指南

**第一步：理解配置系统**
```bash
# 阅读文件
cat cpp_include/core/config.hpp
cat cpp_src/core/config.cpp

# 思考问题:
# 1. EngineConfig 包含哪些配置项？
# 2. 如何序列化和反序列化配置？
# 3. 默认值是如何设置的？
```

**第二步：追踪推理流程**
```bash
# 从 main 函数开始追踪
cat cpp_examples/simple_inference.cpp

# 找到 Engine::generate 的实现
cat cpp_src/core/engine.cpp

# 绘制调用流程图
```

**第三步：分析内存管理**
```bash
# 研究 MemoryManager
cat cpp_include/runtime/memory_manager.hpp
cat cpp_src/runtime/memory_manager.cpp

# 实验：修改内存池大小，观察性能变化
```

### 5.2 动手练习

#### 练习 1: 添加新的激活函数

```cpp
// 在 cpp_include/kernel/activation.hpp 中添加
#pragma once

namespace llm_lv1 {
namespace kernel {

// Swish 激活函数
inline float swish(float x) {
    return x / (1.0f + std::exp(-x));
}

// Mish 激活函数
inline float mish(float x) {
    return x * std::tanh(std::log1p(std::exp(x)));
}

// 批量版本
void swish_activation(float* data, int size);
void mish_activation(float* data, int size);

}  // namespace kernel
}  // namespace llm_lv1
```

```cpp
// 在 cpp_src/kernel/activation.cpp 中实现
#include "kernel/activation.hpp"
#include <cmath>

namespace llm_lv1 {
namespace kernel {

void swish_activation(float* data, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        data[i] = swish(data[i]);
    }
}

void mish_activation(float* data, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        data[i] = mish(data[i]);
    }
}

}  // namespace kernel
}  // namespace llm_lv1
```

#### 练习 2: 实现 KV Cache

```cpp
class KVCache {
private:
    std::vector<float> key_cache_;
    std::vector<float> value_cache_;
    int max_seq_len_;
    int d_model_;
    int current_len_ = 0;
    
public:
    KVCache(int max_seq_len, int d_model)
        : max_seq_len_(max_seq_len), d_model_(d_model) {
        key_cache_.resize(max_seq_len * d_model);
        value_cache_.resize(max_seq_len * d_model);
    }
    
    void update(const float* new_key, const float* new_value, 
                int seq_len) {
        // 追加新的 KV
        std::copy(new_key, new_key + seq_len * d_model_,
                 key_cache_.data() + current_len_ * d_model_);
        std::copy(new_value, new_value + seq_len * d_model_,
                 value_cache_.data() + current_len_ * d_model_);
        current_len_ += seq_len;
    }
    
    const float* get_keys() const { return key_cache_.data(); }
    const float* get_values() const { return value_cache_.data(); }
    int get_length() const { return current_len_; }
    
    void clear() { current_len_ = 0; }
};
```

#### 练习 3: 添加性能分析

```cpp
#include <chrono>

class Profiler {
private:
    std::unordered_map<std::string, 
        std::vector<std::chrono::microseconds>> stats_;
    
public:
    void record(const std::string& name, 
                std::chrono::microseconds duration) {
        stats_[name].push_back(duration);
    }
    
    void print_summary() {
        for (const auto& [name, durations] : stats_) {
            auto total = std::accumulate(durations.begin(), 
                                         durations.end(), 
                                         std::chrono::microseconds(0));
            auto avg = total / durations.size();
            auto min = *std::min_element(durations.begin(), 
                                         durations.end());
            auto max = *std::max_element(durations.begin(), 
                                         durations.end());
            
            std::cout << name << ":\n"
                      << "  Total: " << total.count() << " μs\n"
                      << "  Avg: " << avg.count() << " μs\n"
                      << "  Min: " << min.count() << " μs\n"
                      << "  Max: " << max.count() << " μs\n"
                      << "  Count: " << durations.size() << "\n";
        }
    }
};

// 使用宏简化
#define PROFILE(name) \
    auto _start_##name = std::chrono::high_resolution_clock::now(); \
    struct _Profiler_##name { \
        std::string name_; \
        decltype(_start_##name) start_; \
        _Profiler_##name(std::string n, decltype(_start_##name) s) \
            : name_(n), start_(s) {} \
        ~_Profiler_##name() { \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto duration = std::chrono::duration_cast<\
                std::chrono::microseconds>(end - start_); \
            profiler.record(name_, duration); \
        } \
    } _profiler_##name(#name, _start_##name);
```

---

## 第六部分：调试和优化技巧

### 6.1 常见错误排查

**内存泄漏检测**:
```bash
# 使用 valgrind
valgrind --leak-check=full ./simple_inference

# 使用 AddressSanitizer
g++ -fsanitize=address -g program.cpp
./a.out
```

**性能分析**:
```bash
# 使用 perf
perf record ./simple_inference
perf report

# 使用 gprof
g++ -pg program.cpp
./a.out
gprof a.out gmon.out > profile.txt
```

### 6.2 优化检查清单

- [ ] 启用编译器优化 (-O3)
- [ ] 使用 -march=native
- [ ] 启用 OpenMP
- [ ] 检查内存对齐
- [ ] 减少不必要的拷贝 (使用 move)
- [ ] 预分配容器大小
- [ ] 使用缓存友好的数据布局
- [ ] 避免频繁的内存分配
- [ ] 并行化计算密集型任务

---

## 第七部分：进阶主题

### 7.1 量化技术

```cpp
// INT8 量化
class Quantizer {
public:
    static std::vector<int8_t> quantize_fp32_to_int8(
        const float* input, 
        int size,
        float& scale
    ) {
        // 找到最大值
        float max_val = 0.0f;
        for (int i = 0; i < size; ++i) {
            max_val = std::max(max_val, std::abs(input[i]));
        }
        
        // 计算缩放因子
        scale = max_val / 127.0f;
        
        // 量化
        std::vector<int8_t> quantized(size);
        for (int i = 0; i < size; ++i) {
            quantized[i] = static_cast<int8_t>(
                std::round(input[i] / scale)
            );
        }
        
        return quantized;
    }
    
    static float dequantize_int8_to_fp32(
        int8_t value, 
        float scale
    ) {
        return value * scale;
    }
};
```

### 7.2 算子融合

```cpp
// 融合 MatMul + Add + LayerNorm
void fused_matmul_add_layernorm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int hidden_size,
    float eps = 1e-5f
) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // 1. MatMul + Bias (融合)
        std::vector<float> temp(hidden_size, 0.0f);
        for (int h = 0; h < hidden_size; ++h) {
            temp[h] = bias[h];
            for (int d = 0; d < hidden_size; ++d) {
                temp[h] += input[b * hidden_size + d] * 
                          weight[d * hidden_size + h];
            }
        }
        
        // 2. LayerNorm (融合)
        float mean = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            mean += temp[h];
        }
        mean /= hidden_size;
        
        float variance = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float diff = temp[h] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;
        
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int h = 0; h < hidden_size; ++h) {
            output[b * hidden_size + h] = 
                (temp[h] - mean) * inv_std;
        }
    }
}
```

---

## 总结

通过本教程，你已经学习了：

1. ✅ C++11/14/17 核心特性
2. ✅ 高性能内存管理技术
3. ✅ 并行计算和多线程编程
4. ✅ Transformer 架构的核心组件
5. ✅ 实际项目的开发和优化技巧

**下一步**:
- 贡献代码到项目
- 实现更多算子
- 添加新模型支持
- 参与性能优化

祝你学习愉快！🚀
