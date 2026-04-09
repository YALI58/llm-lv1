# 06-Examples 示例程序教程

本目录包含 LLM-LV1 框架示例程序的详细教学。

## 文件列表

- [demo_showcase.cpp 完全解析](01-demo-showcase.md) - 核心功能演示
- [simple_inference.cpp 解析](02-simple-inference.md) - 简单推理示例
- [streaming_example.cpp 解析](03-streaming-example.md) - 流式生成示例

## 如何学习示例

1. **先运行** - 看到效果再读代码
2. **逐行阅读** - 理解每行代码的作用
3. **修改实验** - 改变参数观察结果
4. **举一反三** - 基于示例构建自己的应用

## 编译和运行

```bash
# 编译所有示例
mkdir -p build && cd build
cmake ..
make

# 运行 demo_showcase
./demo_showcase

# 运行简单推理
./simple_inference

# 运行流式示例
./streaming_example
```
