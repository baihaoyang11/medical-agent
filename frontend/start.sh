#!/bin/bash

# 医学RAG问答系统 - 快速启动脚本
# 用于毕业答辩展示

echo "=========================================="
echo "🎓 医学RAG问答系统 - 前端界面启动脚本"
echo "=========================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python3"
    echo "请先安装 Python3"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📂 工作目录: $SCRIPT_DIR"
echo ""
echo "🚀 正在启动 HTTP 服务器..."
echo ""
echo "✅ 服务器将在以下地址启动："
echo "   - http://localhost:8000"
echo "   - http://127.0.0.1:8000"
echo ""
echo "📖 使用说明："
echo "   1. 确保后端服务已启动（在另一个终端运行 'python agent.py'）"
echo "   2. 在浏览器中打开 http://localhost:8000"
echo "   3. 按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo ""

# 启动 HTTP 服务器
python3 -m http.server 8000
