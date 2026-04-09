#!/bin/bash
# 进程 2: 检查漏洞脚本

echo "[$(date '+%H:%M:%S')] [CHECKER] 开始漏洞检查..." >> logs/workflow.log

CHECK_REPORT="logs/check_report_$(date +%H%M%S).txt"
echo "=== 漏洞检查报告 ===" > "$CHECK_REPORT"
echo "时间：$(date)" >> "$CHECK_REPORT"
echo "" >> "$CHECK_REPORT"

ISSUE_COUNT=0

# 检查 1: 除零风险
if grep -q "hidden_size == 0" cpp_src/kernel/normalization.cpp; then
    echo "✓ hidden_size 零值检查已添加" >> "$CHECK_REPORT"
else
    echo "✗ hidden_size 零值检查缺失" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

# 检查 2: 整数溢出保护
if grep -q "SIZE_MAX" cpp_src/runtime/memory_manager.cpp; then
    echo "✓ memory_manager 中添加了 SIZE_MAX 溢出检查" >> "$CHECK_REPORT"
else
    echo "✗ memory_manager 中缺少 SIZE_MAX 溢出检查" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

if grep -q "SIZE_MAX" cpp_src/kernel/attention.cpp; then
    echo "✓ attention 中添加了 SIZE_MAX 溢出检查" >> "$CHECK_REPORT"
else
    echo "✗ attention 中缺少 SIZE_MAX 溢出检查" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

# 检查 3: 边界检查
if grep -q "page_table\[p\] >= num_pages" cpp_src/kernel/attention.cpp; then
    echo "✓ 页表索引边界检查已添加" >> "$CHECK_REPORT"
else
    echo "✗ 页表索引边界检查缺失" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

if grep -q "copy_size > cache_total_size" cpp_src/kernel/attention.cpp; then
    echo "✓ memcpy 边界检查已添加" >> "$CHECK_REPORT"
else
    echo "✗ memcpy 边界检查缺失" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

# 检查 4: Use-After-Free 修复
if grep -q "allocated_block_indices_" cpp_src/runtime/memory_manager.cpp; then
    echo "✓ 使用索引映射避免迭代器失效" >> "$CHECK_REPORT"
else
    echo "✗ 仍可能存在迭代器失效问题" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

# 检查 5: 空指针检查
NULL_CHECK_COUNT=$(grep -c "nullptr" cpp_src/kernel/attention.cpp || echo 0)
if [ "$NULL_CHECK_COUNT" -gt 3 ]; then
    echo "✓ attention.cpp 中有 $NULL_CHECK_COUNT 处空指针检查" >> "$CHECK_REPORT"
else
    echo "⚠ attention.cpp 中空指针检查较少 ($NULL_CHECK_COUNT 处)" >> "$CHECK_REPORT"
fi

# 检查 6: 参数验证
if grep -q "invalid_argument" cpp_src/kernel/normalization.cpp && \
   grep -q "invalid_argument" cpp_src/runtime/memory_manager.cpp && \
   grep -q "invalid_argument" cpp_src/kernel/attention.cpp; then
    echo "✓ 所有关键文件都有参数验证" >> "$CHECK_REPORT"
else
    echo "✗ 部分文件缺少参数验证" >> "$CHECK_REPORT"
    ((ISSUE_COUNT++))
fi

echo "" >> "$CHECK_REPORT"
echo "=== 检查总结 ===" >> "$CHECK_REPORT"
if [ $ISSUE_COUNT -eq 0 ]; then
    echo "✓ 所有关键安全检查已通过！" >> "$CHECK_REPORT"
    echo "[$(date '+%H:%M:%S')] [CHECKER] ✓ 所有检查通过" >> logs/workflow.log
else
    echo "✗ 发现 $ISSUE_COUNT 个潜在问题" >> "$CHECK_REPORT"
    echo "[$(date '+%H:%M:%S')] [CHECKER] ✗ 发现 $ISSUE_COUNT 个问题" >> logs/workflow.log
fi

cat "$CHECK_REPORT"
echo ""
echo "[$(date '+%H:%M:%S')] [CHECKER] 检查报告已保存至：$CHECK_REPORT" >> logs/workflow.log
