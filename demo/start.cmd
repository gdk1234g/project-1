@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "DEMO_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "DEMO_PYTHON=.venv\Scripts\python.exe"
"%DEMO_PYTHON%" -c "import torch, cv2, fastapi, uvicorn, multipart, PIL"
if errorlevel 1 goto missing
echo.
echo 启动叶析本地演示。浏览器访问 http://127.0.0.1:8000
echo 保持本窗口运行，按 Ctrl+C 停止服务。
echo.
"%DEMO_PYTHON%" -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
pause
exit /b
:missing
echo.
echo 演示环境缺少依赖，请按照 README.md 安装后重试。
pause
exit /b 1
