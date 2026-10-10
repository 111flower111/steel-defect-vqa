# -*- coding: utf-8 -*-
"""
钢材缺陷检测报告系统 —— vLLM 部署版 FastAPI 封装

前置：先启动 vLLM 服务（见 start.sh）：
  vllm serve ../quantize/awq-int4 --host 0.0.0.0 --port 8000 --served-model-name steel-vqa-awq

本服务负责：接收图片上传 → 转 base64 → 调用 vLLM → 返回结构化报告

用法:
  cd deploy
  python -m uvicorn main_vllm:app --host 0.0.0.0 --port 8001
"""
import base64
import io
import json
import os

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "steel-vqa-awq"  # vLLM 里注册的模型名（AWQ INT4 量化模型）

QUESTION = ("请检测这张钢材表面图像中是否存在缺陷。如有，"
            "请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。")

app = FastAPI(title="钢材缺陷检测报告系统 (vLLM)")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/inspect")
async def inspect(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": QUESTION},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(VLLM_URL, json=payload)
        data = resp.json()

    # 提取 vLLM 返回的文本
    try:
        raw = data["choices"][0]["message"]["content"]
    except Exception:
        raw = json.dumps(data, ensure_ascii=False)

    report = extract_json(raw)
    return JSONResponse({"report": report, "raw": raw})


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None
