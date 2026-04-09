# 03-Runtime 运行时教程

本目录包含 LLM-LV1 框架运行时组件的详细教学文档。

## 文件列表

- [memory_manager.hpp 详解](01-runtime-memory-manager.md) - 内存管理完全指南
- [scheduler.cpp 解析](02-runtime-scheduler.md) - 任务调度器
- [context.cpp 解析](03-runtime-context.md) - 执行上下文

## 什么是 Runtime？

**Runtime（运行时）** 是程序实际执行时的环境和管理系统：
- **内存管理**: 分配、回收、优化
- **任务调度**: 并发控制、资源分配
- **执行上下文**: 状态跟踪、错误处理

## 学习重点

1. **内存池化技术** - 减少分配开销
2. **分页注意力** - vLLM 核心优化
3. **并发调度** - 多请求处理
