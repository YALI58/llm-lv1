# LLM-LV1 完整学习路线

欢迎来到 LLM-LV1 框架的学习之旅！本指南将带你从零开始，逐步掌握大型语言模型推理引擎的开发。

---

## 📚 学习路径总览

```
第 1 阶段：基础认知 (1-2 天)
    └─> 了解项目结构和核心概念

第 2 阶段：核心模块 (3-5 天)
    └─> 深入学习配置、状态、引擎

第 3 阶段：计算 Kernel (2-3 天)
    └─> 掌握注意力机制和优化技术

第 4 阶段：运行时系统 (2-3 天)
    └─> 理解内存管理和调度

第 5 阶段：实战应用 (3-5 天)
    └─> 构建自己的应用
```

---

## 🎯 第 1 阶段：基础认知

### 目标
- 了解项目整体结构
- 理解 LLM 推理的基本流程
- 搭建开发环境

### 学习内容

1. **阅读 README.md**
   - 项目介绍
   - 特性列表
   - 快速开始

2. **运行 demo_showcase**
   ```bash
   mkdir -p build && cd build
   cmake ..
   make demo_showcase
   ./demo_showcase
   ```

3. **浏览目录结构**
   ```
   cpp_include/     # 头文件
   cpp_src/         # 源文件
   cpp_examples/    # 示例程序
   cpp_tests/       # 单元测试
   ```

### 检查清单
- [ ] 成功编译项目
- [ ] 运行 demo_showcase 看到输出
- [ ] 理解每个目录的作用

---

## 🎯 第 2 阶段：核心模块

### 目标
- 掌握配置系统
- 理解状态管理
- 学会使用推理引擎

### 学习内容

#### Day 1: 配置系统
- 📖 阅读 [01-core-config.md](tutorial/01-core/01-core-config.md)
- 🔧 练习：创建不同场景的配置
- ✅ 完成文档中的练习题

#### Day 2: 状态管理
- 📖 阅读 [03-core-state.md](tutorial/01-core/03-core-state.md)
- 🔧 练习：实现自定义停止条件
- ✅ 理解 GenerationState 的生命周期

#### Day 3: 推理引擎
- 📖 阅读 [02-core-engine.md](tutorial/01-core/02-core-engine.md)
- 🔧 练习：编写简单的对话机器人
- ✅ 掌握 generate() 和 generate_stream()

### 检查清单
- [ ] 能创建适合不同场景的配置
- [ ] 理解 KV Cache 的作用
- [ ] 能编写基本的文本生成程序

---

## 🎯 第 3 阶段：计算 Kernel

### 目标
- 深入理解注意力机制
- 了解优化技术原理

### 学习内容

#### Day 1: 注意力机制基础
- 📖 阅读 [01-kernel-attention.md](tutorial/02-kernel/01-kernel-attention.md)
- 📐 复习 Transformer 的注意力公式
- 🔧 练习：手动计算简单注意力

#### Day 2: Paged Attention
- 📖 深入学习分页注意力原理
- 📊 对比传统方式和分页方式的内存使用
- 🔧 练习：调整 page_size 观察效果

#### Day 3: 性能优化
- 📖 了解 Flash Attention 原理
- 📈 学习性能分析方法
- 🔧 练习：批量大小对性能的影响

### 检查清单
- [ ] 能解释注意力机制的工作原理
- [ ] 理解 Paged Attention 的优势
- [ ] 能进行简单的性能测试

---

## 🎯 第 4 阶段：运行时系统

### 目标
- 掌握内存管理技术
- 理解任务调度原理

### 学习内容

#### Day 1: 内存管理器
- 📖 阅读 [01-runtime-memory-manager.md](tutorial/03-runtime/01-runtime-memory-manager.md)
- 🔧 练习：实现简单内存池
- 📊 分析内存使用统计

#### Day 2: 任务调度
- 📖 阅读 scheduler 相关代码
- 🔧 练习：多请求并发处理
- 📈 分析调度策略

#### Day 3: 综合实践
- 🔧 结合内存管理和调度
- 🏗️ 设计一个简单的推理服务

### 检查清单
- [ ] 理解内存池化的优势
- [ ] 能分配和释放 KV Cache
- [ ] 能监控内存使用情况

---

## 🎯 第 5 阶段：实战应用

### 目标
- 构建完整的應用
- 解决实际问题

### 项目建议

#### 项目 1: 命令行聊天机器人
```cpp
// 功能要求
- 循环读取用户输入
- 流式输出生成结果
- 支持退出命令
- 保存对话历史
```

#### 项目 2: 批量文本生成器
```cpp
// 功能要求
- 从文件读取 prompts
- 批量生成回复
- 进度显示
- 结果保存到文件
```

#### 项目 3: API 服务
```cpp
// 功能要求
- HTTP REST API
- 并发请求处理
- 请求队列管理
- 性能监控
```

### 检查清单
- [ ] 完成至少一个实战项目
- [ ] 代码有清晰的注释
- [ ] 能处理错误情况

---

## 📖 推荐学习资源

### 必读论文
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
2. [vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180) - Paged Attention 论文

### 在线课程
- [Hugging Face NLP Course](https://huggingface.co/course)
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/)

### 技术博客
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

---

## 🤝 学习社区

### 提问渠道
- GitHub Issues: 报告 bug 和提出功能请求
- 项目讨论区：交流学习问题

### 贡献方式
- 提交文档改进
- 分享学习心得
- 贡献代码优化

---

## 📝 学习日记模板

建议每天记录学习进度：

```markdown
## Day X - YYYY-MM-DD

### 今天学了什么
- 

### 遇到的问题
- 

### 解决方案
- 

### 明天的计划
- 
```

---

## 🎓 结业考核

完成以下任务证明你掌握了 LLM-LV1：

### 基础题（必做）
1. 编写一个对话机器人
2. 实现自定义的采样策略
3. 分析并优化内存使用

### 进阶题（选做）
1. 添加新的 kernel 实现
2. 实现连续批处理
3. 添加 GPU 支持

### 挑战题（加分）
1. 实现 speculative decoding
2. 集成多种量化方法
3. 构建分布式推理系统

---

## 📅 学习时间规划

| 阶段 | 建议时间 | 累计时间 |
|------|---------|---------|
| 基础认知 | 1-2 天 | 2 天 |
| 核心模块 | 3-5 天 | 7 天 |
| 计算 Kernel | 2-3 天 | 10 天 |
| 运行时系统 | 2-3 天 | 13 天 |
| 实战应用 | 3-5 天 | 18 天 |

**总计**: 约 2-3 周（每天 2-3 小时）

---

## 💡 学习建议

1. **循序渐进**: 不要跳过一个阶段
2. **动手实践**: 光看不练学不会
3. **善用搜索**: 遇到问题先搜索
4. **记录笔记**: 好记性不如烂笔头
5. **分享交流**: 教是最好的学

---

## 🚀 下一步

现在就开始你的学习之旅吧！

👉 从 [README.md](../README.md) 开始，然后运行 `demo_showcase`！

祝学习愉快！🎉

---

*最后更新：2024 | LLM-LV1 教学文档系列*
