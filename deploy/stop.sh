#!/bin/bash
# 停止钢材缺陷检测系统服务
source /home/wxy/anaconda3/etc/profile.d/conda.sh

echo "停止 vLLM ..."
kill $(pgrep -f "vllm serve" 2>/dev/null) 2>/dev/null || true

echo "停止 FastAPI ..."
kill $(pgrep -f "uvicorn main_vllm" 2>/dev/null) 2>/dev/null || true

echo "✅ 服务已停止"
