#!/bin/bash
# ============================================
# 钢材缺陷检测系统 —— 一键启动脚本（vLLM 版）
# ============================================
# 启动两个服务：
#   1. vLLM 推理引擎（端口 8000，OpenAI 兼容 API）
#   2. FastAPI 封装 + 网页前端（端口 8001）
#
# 用法: bash start.sh
# 访问: http://127.0.0.1:8001
# ============================================

set -e
source /home/wxy/anaconda3/etc/profile.d/conda.sh
ROOT=/home/wxy/factory_test/steel-defect-vqa

echo "=== 1/2 启动 vLLM 推理引擎（vllm-deploy 环境，AWQ INT4 量化模型）==="
conda activate vllm-deploy
cd "$ROOT"
nohup vllm serve quantize/awq-int4 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name steel-vqa-awq \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5 \
    > deploy/vllm_awq.log 2>&1 &

echo "=== 2/2 启动 FastAPI 封装 + 前端（steel-vqa 环境）==="
conda activate steel-vqa
cd "$ROOT/deploy"
nohup python -m uvicorn main_vllm:app --host 0.0.0.0 --port 8001 \
    > fastapi_vllm.log 2>&1 &

echo ""
echo "✅ 服务启动中，等待约 30~60 秒（vLLM 加载模型）..."
echo "   前端页面: http://127.0.0.1:8001"
echo "   vLLM API: http://127.0.0.1:8000"
echo "   停止服务: bash stop.sh"
