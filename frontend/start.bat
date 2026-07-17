@echo off
chcp 65001 >nul
echo ==========================================
echo 🎓 医学RAG问答系统 - 前端界面启动脚本
echo ==========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python
    echo 请先安装 Python
    pause
    exit /b 1
)

echo 📂 工作目录: %~dp0
echo.
echo 🚀 正在启动 HTTP 服务器...
echo.
echo ✅ 服务器将在以下地址启动：
echo    - http://localhost:8000
echo    - http://127.0.0.1:8000
echo.
echo 📖 使用说明：
echo    1. 确保后端服务已启动（在另一个命令行窗口运行 'python agent.py'）
echo    2. 在浏览器中打开 http://localhost:8000
echo    3. 按 Ctrl+C 停止服务器
echo.
echo ==========================================
echo.

REM 启动 HTTP 服务器
python -m http.server 8000

pause
