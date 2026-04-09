# LLM-LV1 教学文档索引

欢迎使用 LLM-LV1 框架教学文档！本索引帮助你快速找到所需的学习资料。

---

## 📚 文档结构

```
docs/tutorial/
├── LEARNING_PATH.md           # 完整学习路线（从这里开始！）
├── 01-core/                   # 核心模块教程
│   ├── README.md
│   ├── 01-core-config.md      # 配置系统详解
│   ├── 02-core-engine.md      # 推理引擎详解
│   └── 03-core-state.md       # 状态管理详解
├── 02-kernel/                 # 计算层教程
│   ├── README.md
│   ├── 01-kernel-attention.md # 注意力机制详解
│   ├── 02-kernel-activation.md
│   └── 03-kernel-normalization.md
├── 03-runtime/                # 运行时教程
│   ├── README.md
│   ├── 01-runtime-memory-manager.md  # 内存管理详解
│   ├── 02-runtime-scheduler.md
│   └── 03-runtime-context.md
├── 04-model/                  # 模型层教程
│   └── README.md
├── 05-utils/                  # 工具模块教程
│   └── README.md
└── 06-examples/               # 示例程序教程
    ├── README.md
    ├── 01-demo-showcase.md    # demo_showcase 完全解析
    ├── 02-simple-inference.md
    └── 03-streaming-example.md
```

---

## 🚀 快速开始

### 新手入门路径

1. **第一步**: 阅读 [LEARNING_PATH.md](LEARNING_PATH.md) - 了解完整学习路线
2. **第二步**: 运行 `demo_showcase` - 查看框架效果
3. **第三步**: 阅读 [01-core-config.md](01-core/01-core-config.md) - 学习配置系统
4. **第四步**: 阅读 [02-core-engine.md](01-core/02-core-engine.md) - 学习推理引擎
5. **第五步**: 完成实战项目 - 构建自己的应用

### 按主题查找

#### 我想学习配置系统
→ [01-core-config.md](01-core/01-core-config.md)

#### 我想学习如何使用推理引擎
→ [02-core-engine.md](01-core/02-core-engine.md)

#### 我想理解注意力机制
→ [01-kernel-attention.md](02-kernel/01-kernel-attention.md)

#### 我想学习内存管理
→ [01-runtime-memory-manager.md](03-runtime/01-runtime-memory-manager.md)

#### 我想看示例代码
→ [01-demo-showcase.md](06-examples/01-demo-showcase.md)

---

## 📖 文档详情

### 01-Core 核心模块

| 文档 | 内容 | 难度 | 预计时间 |
|------|------|------|---------|
| [config.md](01-core/01-core-config.md) | 设备、量化、内存配置 | ⭐⭐ | 2 小时 |
| [engine.md](01-core/02-core-engine.md) | 推理引擎 API 使用 | ⭐⭐ | 2 小时 |
| [state.md](01-core/03-core-state.md) | 生成状态管理 | ⭐⭐⭐ | 1 小时 |

### 02-Kernel 计算层

| 文档 | 内容 | 难度 | 预计时间 |
|------|------|------|---------|
| [attention.md](02-kernel/01-kernel-attention.md) | 注意力机制实现 | ⭐⭐⭐⭐ | 3 小时 |

### 03-Runtime 运行时

| 文档 | 内容 | 难度 | 预计时间 |
|------|------|------|---------|
| [memory-manager.md](03-runtime/01-runtime-memory-manager.md) | 内存池化和 KV Cache | ⭐⭐⭐ | 2 小时 |

### 06-Examples 示例程序

| 文档 | 内容 | 难度 | 预计时间 |
|------|------|------|---------|
| [demo-showcase.md](06-examples/01-demo-showcase.md) | 综合功能演示 | ⭐⭐ | 2 小时 |

---

## 🎯 学习检查清单

### 基础级（必须掌握）
- [ ] 能创建 EngineConfig 配置
- [ ] 能使用 InferenceEngine 生成文本
- [ ] 理解 KV Cache 的作用
- [ ] 能运行 demo_showcase

### 进阶级（推荐掌握）
- [ ] 理解 Paged Attention 原理
- [ ] 能调整配置优化性能
- [ ] 能实现流式输出
- [ ] 能监控内存使用

### 专家级（挑战目标）
- [ ] 能实现自定义 kernel
- [ ] 能优化内存管理策略
- [ ] 能实现并发调度
- [ ] 能构建完整服务

---

## 🔗 相关资源

### 项目文档
- [README.md](../README.md) - 项目介绍
- [LEARNING_GUIDE.md](../LEARNING_GUIDE.md) - 学习指南

### 外部资源
- Transformer 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- vLLM 论文：[vLLM: Easy, Fast, and Cheap LLM Serving](https://arxiv.org/abs/2309.06180)
- Hugging Face 课程：[NLP Course](https://huggingface.co/course)

---

## 📝 文档统计

- **总文档数**: 13 个
- **总行数**: 约 3000 行
- **覆盖范围**: 配置、引擎、注意力、内存管理、示例程序
- **适合人群**: 初学者到进阶开发者

---

## 💡 使用建议

1. **按顺序学习**: 从 LEARNING_PATH.md 开始，按阶段学习
2. **边学边练**: 每个文档都有练习题，务必完成
3. **记录笔记**: 使用提供的学习日记模板
4. **分享交流**: 遇到问题多提问，学会后多分享

---

## 🤝 贡献文档

发现文档问题或有改进建议？欢迎贡献！

### 可以贡献的内容
- 修正错别字和错误
- 补充遗漏的知识点
- 添加新的示例
- 翻译其他语言版本

### 如何贡献
1. Fork 项目
2. 修改文档
3. 提交 Pull Request

---

*最后更新：2024 | LLM-LV1 教学文档系列*
