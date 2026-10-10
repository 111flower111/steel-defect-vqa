# -*- coding: utf-8 -*-
"""
AWQ 量化模型精度评估（通过 vLLM API）

在验证集上抽样，调用 vLLM 的 OpenAI 兼容接口，统计缺陷分类准确率 + JSON 合规率。

用法:
  python evaluate_awq.py --max-samples 200 --model steel-vqa-awq
"""
import argparse
import base64
import json
import os
import random
import urllib.request
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
QUESTION = ("请检测这张钢材表面图像中是否存在缺陷。如有，"
            "请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。")


def extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--model", default="steel-vqa-awq")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 读验证集，按图去重（每图取第一个"检测+JSON"问题）
    samples = {}
    with open(os.path.join(BASE, "data", "steel_vqa_val.jsonl"), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if not s["id"].endswith("_0"):
                continue
            gt = json.loads(s["conversations"][1]["value"])
            samples[s["image"]] = {"path": os.path.join(BASE, "data", s["image"]),
                                   "gt": gt["defect_type"]}
    keys = list(samples.keys())
    random.seed(args.seed)
    keys = random.sample(keys, min(args.max_samples, len(keys)))
    print(f"评估 {len(keys)} 张图 ...", flush=True)

    correct = compliant = 0
    per_class = defaultdict(lambda: [0, 0])
    for i, k in enumerate(keys):
        s = samples[k]
        gt = s["gt"]
        per_class[gt][1] += 1
        with open(s["path"], "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": QUESTION},
            ]}],
            "max_tokens": 256, "temperature": 0,
        }
        try:
            req = urllib.request.Request(VLLM_URL, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
            out = resp["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [{i+1}] 请求失败: {e}", flush=True)
            continue
        parsed = extract_json(out)
        if parsed and "defect_type" in parsed:
            compliant += 1
            if parsed["defect_type"] == gt:
                correct += 1
                per_class[gt][0] += 1
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(keys)} | 当前准确率 {correct}/{i+1}", flush=True)

    total = len(keys)
    print("\n" + "=" * 50)
    print(f"总样本数: {total}")
    print(f"JSON 合规率: {compliant}/{total} = {compliant/total*100:.1f}%")
    print(f"缺陷分类准确率: {correct}/{total} = {correct/total*100:.1f}%")
    print("\n=== 各类别准确率（抽样>=3）===")
    for cls, (c, t) in sorted(per_class.items(), key=lambda x: -x[1][1]):
        if t >= 3:
            print(f"  {cls:32s} {c:3d}/{t:3d} = {c/t*100:5.1f}%")


if __name__ == "__main__":
    main()
