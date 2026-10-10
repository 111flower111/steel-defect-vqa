# -*- coding: utf-8 -*-
"""
训练后验证脚本（阶段三 Step 3.3，标准 transformers + peft 方案）

加载训练好的 LoRA，对一张图片推理，检查输出是否为结构化 JSON。

用法:
  python test_inference.py --image ../data/images/val/bs_05.jpg
"""
import argparse
import os

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_PATH = os.path.join(os.path.dirname(__file__), "outputs", "steel-vqa-lora")

QUESTION = ("请检测这张钢材表面图像中是否存在缺陷。如有，"
            "请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--lora", default=LORA_PATH)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.lora)  # 挂载 LoRA
    processor = AutoProcessor.from_pretrained(args.lora)
    model = model.to("cuda").eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(new_ids, skip_special_tokens=True)
    print("=== 模型输出 ===")
    print(answer)


if __name__ == "__main__":
    main()
