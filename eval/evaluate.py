# -*- coding: utf-8 -*-
"""
模型评估脚本（阶段六）

在验证集上批量推理，统计：
  1. JSON 合规率（输出能被 json.loads 解析的比例）
  2. 缺陷分类准确率（模型 defect_type 与真实标签一致的比例）

用法:
  # 快速评估（随机抽 400 张）
  python evaluate.py --max-samples 400

  # 全量评估（2324 张，约 2 小时）
  python evaluate.py
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_LORA = os.path.join(os.path.dirname(__file__), "..", "train", "outputs", "steel-vqa-lora")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

QUESTION = ("请检测这张钢材表面图像中是否存在缺陷。如有，"
            "请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。")


def extract_json(text):
    """从模型输出中提取 JSON 对象（容错：忽略前后多余文字）"""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def load_model(lora_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, lora_path)
    processor = AutoProcessor.from_pretrained(lora_path)
    model = model.to("cuda").eval()
    return model, processor


def predict(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--max-samples", type=int, default=0, help="0=全量, >0=随机抽 N 张")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. 读验证集，按图去重（每张图 3 个问题，只取第一个"检测+JSON"问题）
    samples = {}
    with open(os.path.join(args.data_dir, "steel_vqa_val.jsonl"), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if not s["id"].endswith("_0"):
                continue
            gt = json.loads(s["conversations"][1]["value"])
            samples[s["image"]] = {
                "image_path": os.path.join(args.data_dir, s["image"]),
                "gt": gt["defect_type"],
            }
    print(f"验证集图片数: {len(samples)}")

    keys = list(samples.keys())
    if args.max_samples > 0:
        random.seed(args.seed)
        keys = random.sample(keys, min(args.max_samples, len(keys)))
        print(f"本次评估抽样: {len(keys)} 张")

    # 2. 加载模型
    model, processor = load_model(args.lora)

    # 3. 批量推理
    correct = 0
    compliant = 0
    per_class = defaultdict(lambda: [0, 0])  # class -> [correct, total]
    errors = []

    for img_key in tqdm(keys, desc="评估中"):
        s = samples[img_key]
        gt = s["gt"]
        per_class[gt][1] += 1
        try:
            output = predict(model, processor, s["image_path"])
        except Exception as e:
            errors.append((img_key, gt, f"推理异常:{e}"))
            continue
        parsed = extract_json(output)
        if parsed and "defect_type" in parsed:
            compliant += 1
            pred = parsed["defect_type"]
            if pred == gt:
                correct += 1
                per_class[gt][0] += 1
            else:
                errors.append((img_key, gt, pred))
        else:
            errors.append((img_key, gt, "非JSON输出"))

    total = len(keys)
    print("\n" + "=" * 50)
    print(f"总样本数: {total}")
    print(f"JSON 合规率: {compliant}/{total} = {compliant/total*100:.1f}%")
    print(f"缺陷分类准确率: {correct}/{total} = {correct/total*100:.1f}%")
    if compliant > 0:
        print(f"（合规样本内准确率: {correct}/{compliant} = {correct/compliant*100:.1f}%）")

    # 4. 各类别准确率（仅统计抽样>=3的类）
    print("\n=== 各类别准确率（抽样数>=3）===")
    for cls, (c, t) in sorted(per_class.items(), key=lambda x: -x[1][1]):
        if t >= 3:
            print(f"  {cls:32s} {c:3d}/{t:3d} = {c/t*100:5.1f}%")

    # 5. 错误样例
    print(f"\n=== 错误样例（前 10 条）===")
    for img, gt, pred in errors[:10]:
        print(f"  {img}: 真实={gt}  预测={pred}")


if __name__ == "__main__":
    main()
