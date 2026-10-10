# -*- coding: utf-8 -*-
"""
SteelDefectX 数据集探索脚本（阶段一）

功能：
  1. 从 Hugging Face 下载 train-text.json / val-text.json / class_descriptions.json
  2. 打印字段结构与样例
  3. 统计 25 类缺陷分布与不均衡度
  4. 分析 structured_attributes 的取值空间（用于阶段二构造 VQA 答案）
  5. 核对 image_name 前缀与类别映射

用法：
  export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
  python explore_data.py
"""
import json
import re
import os
from collections import Counter, defaultdict

import requests

REPO = "Zhaosxian/SteelDefectX"
BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
OUT_DIR = os.path.join(os.path.dirname(__file__), "meta")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = ["train-text.json", "val-text.json", "class_descriptions.json"]


def download(name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        url = f"{BASE}/datasets/{REPO}/resolve/main/{name}"
        print(f"下载 {name} ...")
        r = requests.get(url, allow_redirects=True, timeout=300)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
    return path


def main():
    rows = []
    for name in ["train-text.json", "val-text.json"]:
        split = "train" if name.startswith("train") else "val"
        with open(download(name), encoding="utf-8") as f:
            data = json.load(f)
        print(f"[{name}] 样本数 = {len(data)}")
        for r in data:
            r["_split"] = split
        rows.extend(data)

    print(f"\n总样本数 = {len(rows)}")

    # 1. 字段结构
    print("\n=== 每条样本的字段 ===")
    for k in rows[0].keys():
        v = rows[0][k]
        print(f"  {k}: {type(v).__name__}")

    # 2. 类别分布
    cls = Counter(r["class_name"] for r in rows)
    print("\n=== 25 类缺陷分布 (train / val) ===")
    for c, n in cls.most_common():
        tr = sum(1 for r in rows if r["class_name"] == c and r["_split"] == "train")
        va = sum(1 for r in rows if r["class_name"] == c and r["_split"] == "val")
        print(f"  {c:32s} 总{n:4d}  train {tr:4d} / val {va:4d}")

    counts = list(cls.values())
    print(f"\n类别数={len(counts)}  最多={max(counts)}  最少={min(counts)}  "
          f"中位数={sorted(counts)[len(counts)//2]}  不均衡比={max(counts)/min(counts):.1f}x")

    # 3. structured_attributes 取值空间
    print("\n=== structured_attributes 各字段取值分布 ===")
    attr = defaultdict(Counter)
    for r in rows:
        for k, v in r.get("structured_attributes", {}).items():
            attr[k][str(v) if not isinstance(v, list) else "<list>"] += 1
    for k in sorted(attr):
        top = ", ".join(f"{v}:{c}" for v, c in attr[k].most_common(12))
        print(f"  [{k}] {top}")

    # 4. image_name 前缀 -> class 映射
    print("\n=== image_name 前缀 -> class_name 映射 ===")
    p2c = defaultdict(set)
    for r in rows:
        m = re.match(r"^([A-Za-z]+)", r["image_name"])
        if m:
            p2c[m.group(1)].add(r["class_name"])
    for p in sorted(p2c):
        print(f"  {p:8s} -> {list(p2c[p])[0]}")

    # 5. 类别描述
    with open(download("class_descriptions.json"), encoding="utf-8") as f:
        cdesc = json.load(f)
    print(f"\nclass_descriptions.json 共 {len(cdesc)} 个类别描述")


if __name__ == "__main__":
    main()
