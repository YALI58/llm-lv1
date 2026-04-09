# 02-Kernel 核心计算层教程

本目录包含 LLM-LV1 框架核心计算 kernel 的详细教学文档。

## 文件列表

- [attention.hpp 详解](01-kernel-attention.md) - 注意力机制完全指南
- [activation.cpp 解析](02-kernel-activation.md) - 激活函数实现
- [normalization.cpp 解析](03-kernel-normalization.md) - 归一化层实现

## 什么是 Kernel？

在深度学习中，**Kernel** 指的是最基本的计算单元：
- **Attention Kernel**: 计算注意力权重
- **Activation Kernel**: 应用激活函数（ReLU, SiLU, GELU）
- **Normalization Kernel**: LayerNorm, RMSNorm

## 学习重点

1. **理解数学原理** - 每个 kernel 背后的公式
2. **掌握实现细节** - 如何高效实现这些计算
3. **性能优化** - 内存访问模式、并行化策略

## 前置知识

- 线性代数基础（矩阵乘法、向量运算）
- Transformer 架构基础
- C++ 编程基础
