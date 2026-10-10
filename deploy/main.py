# -*- coding: utf-8 -*-
"""
钢材缺陷检测报告系统 —— FastAPI 服务（部署）

加载合并后的 Qwen2.5-VL 模型，提供图片上传 → 结构化报告接口。

用法:
  cd deploy
  uvicorn main:app --host 0.0.0.0 --port 8001
  # 浏览器打开 http://127.0.0.1:8001
"""
import io
import json
import os

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_MODEL = os.path.join(BASE_DIR, "..", "train", "outputs", "steel-vqa-merged")

QUESTION = ("请检测这张钢材表面图像中是否存在缺陷。如有，"
            "请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。")

app = FastAPI(title="钢材缺陷检测报告系统")

# 启动时加载模型（模块级，只加载一次）
print("加载模型中 ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MERGED_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MERGED_MODEL)
print("✅ 模型加载完成")


def extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/inspect")
async def inspect(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    answer = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    report = extract_json(answer)
    return JSONResponse({"report": report, "raw": answer})
