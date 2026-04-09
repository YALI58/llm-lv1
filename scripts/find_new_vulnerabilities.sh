#!/bin/bash
# 进程 3: 继续查找新漏洞脚本

echo "[$(date '+%H:%M:%S')] [FINDER] 开始深度漏洞扫描..." >> logs/workflow.log

FIND_REPORT="logs/find_report_$(date +%H%M%S).txt"
echo "=== 新漏洞发现报告 ===" > "$FIND_REPORT"
echo "时间：$(date)" >> "$FIND_REPORT"
echo "" >> "$FIND_REPORT"

NEW_ISSUES=0

# 扫描 1: 检查未初始化的指针
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：未初始化指针..." >> logs/workflow.log
UNINIT_PTR=$(grep -rn "= nullptr" cpp_src/ | grep -v "check" | wc -l)
if [ "$UNINIT_PTR" -gt 0 ]; then
    echo "⚠ 发现 $UNINIT_PTR 处指针初始化为 nullptr (需检查是否正确使用)" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
fi

# 扫描 2: 检查裸 new/delete
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：裸 new/delete..." >> logs/workflow.log
RAW_NEW=$(grep -rn "\\bnew\\b" cpp_src/ | grep -v "unique_ptr\|shared_ptr\|make_unique" | wc -l)
RAW_DELETE=$(grep -rn "\\bdelete\\b" cpp_src/ | wc -l)
if [ "$RAW_NEW" -gt 0 ] || [ "$RAW_DELETE" -gt 0 ]; then
    echo "⚠ 发现 $RAW_NEW 处裸 new 和 $RAW_DELETE 处裸 delete (建议使用智能指针)" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
fi

# 扫描 3: 检查数组访问
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：潜在数组越界..." >> logs/workflow.log
ARRAY_ACCESS=$(grep -rn "\[.*\]" cpp_src/kernel/*.cpp | grep -v "size_t\|const\|//" | wc -l)
echo "ℹ 发现 $ARRAY_ACCESS 处数组访问 (需人工审查边界检查)" >> "$FIND_REPORT"

# 扫描 4: 检查类型转换
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：危险类型转换..." >> logs/workflow.log
REINTERPRET=$(grep -rn "reinterpret_cast" cpp_src/ | wc -l)
STATIC_CAST_PTR=$(grep -rn "static_cast<.*\*" cpp_src/ | wc -l)
if [ "$REINTERPRET" -gt 0 ]; then
    echo "⚠ 发现 $REINTERPRET 处 reinterpret_cast (可能存在类型安全问题)" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
fi

# 扫描 5: 检查锁的使用
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：线程安全..." >> logs/workflow.log
LOCK_GUARD=$(grep -rn "lock_guard" cpp_src/ | wc -l)
RAW_LOCK=$(grep -rn "\.lock()\|\.unlock()" cpp_src/ | wc -l)
if [ "$RAW_LOCK" -gt 0 ]; then
    echo "⚠ 发现 $RAW_LOCK 处手动 lock/unlock (建议使用 lock_guard)" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
else
    echo "✓ 所有锁操作都使用 lock_guard ($LOCK_GUARD 处)" >> "$FIND_REPORT"
fi

# 扫描 6: 检查异常处理
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：异常处理..." >> logs/workflow.log
THROW_COUNT=$(grep -rn "throw " cpp_src/ | wc -l)
TRY_COUNT=$(grep -rn "try {" cpp_src/ | wc -l)
echo "ℹ 发现 $THROW_COUNT 处 throw 和 $TRY_COUNT 处 try-catch" >> "$FIND_REPORT"

# 扫描 7: 检查魔法数字
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：魔法数字..." >> logs/workflow.log
MAGIC_NUMS=$(grep -rn "[0-9]\{4,\}" cpp_src/ | grep -v "//\|constexpr" | head -5)
if [ -n "$MAGIC_NUMS" ]; then
    echo "⚠ 发现魔法数字 (建议定义为常量):" >> "$FIND_REPORT"
    echo "$MAGIC_NUMS" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
fi

# 扫描 8: 检查内存拷贝
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：内存拷贝操作..." >> logs/workflow.log
MEMCPY=$(grep -rn "memcpy\|memset\|memmove" cpp_src/ | wc -l)
echo "ℹ 发现 $MEMCPY 处内存拷贝操作 (需确保边界检查)" >> "$FIND_REPORT"

# 扫描 9: 检查浮点比较
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：浮点直接比较..." >> logs/workflow.log
FLOAT_CMP=$(grep -rn "== 0\.0f\|== 0\.\|!= 0\.0f\|!= 0\." cpp_src/ | wc -l)
if [ "$FLOAT_CMP" -gt 0 ]; then
    echo "⚠ 发现 $FLOAT_CMP 处浮点数直接比较 (建议使用 epsilon 比较)" >> "$FIND_REPORT"
    ((NEW_ISSUES++))
fi

# 扫描 10: 检查递归深度
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描：递归函数..." >> logs/workflow.log
RECURSIVE=$(grep -rn "void.*\(.*\).*{" cpp_src/ | grep -A5 "$(grep -l "recursive\|recurse" cpp_src/*/*.hpp 2>/dev/null)" | wc -l)
echo "ℹ 递归函数检查完成" >> "$FIND_REPORT"

echo "" >> "$FIND_REPORT"
echo "=== 扫描总结 ===" >> "$FIND_REPORT"
echo "发现 $NEW_ISSUES 个需要关注的问题" >> "$FIND_REPORT"
echo "详细代码位置请查看上述输出" >> "$FIND_REPORT"

cat "$FIND_REPORT"
echo ""
echo "[$(date '+%H:%M:%S')] [FINDER] 扫描报告已保存至：$FIND_REPORT" >> logs/workflow.log
