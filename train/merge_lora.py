# -*- coding: utf-8 -*-
"""
LoRA 权重合并脚本（部署第一步）

把训练好的 LoRA 适配器合并回基座模型，得到一个完整的可部署模型。

用法:
  python merge_lora.py                     # 合并增强版(94.0%)
  python merge_lora.py --lora ./outputs/steel-vqa-lora --out ./outputs/steel-vqa-merged-v1
"""
import argparse
import os

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_LORA = os.path.join(os.path.dirname(__file__), "outputs", "steel-vqa-lora-aug")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "outputs", "steel-vqa-merged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--base", default=MODEL_NAME)
    args = ap.parse_args()

    print(f"加载基座模型: {args.base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    print(f"加载 LoRA: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora)

    print("合并 LoRA 到基座模型 ...")
    model = model.merge_and_unload()  # 关键：把 LoRA 融进基座权重

    print(f"保存合并后模型到: {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)

    # 保存 processor（分词器 + 图像处理器）
    processor = AutoProcessor.from_pretrained(args.lora)
    processor.save_pretrained(args.out)

    print("✅ 合并完成！模型已保存，可直接用于部署")


if __name__ == "__main__":
    main()
