# -*- coding: utf-8 -*-
"""
SteelDefectX → VQA SFT 数据集构造脚本（阶段二）

依赖（已在 .venv 中）: datasets, pillow, pandas, tqdm, huggingface_hub, requests

真实数据字段（来自阶段一探索，切勿用方案里的 label/coarse_text/fine_text）:
  image_name                       图片文件名（train/xxx.jpg 或 val/xxx.jpg）
  class_name                       缺陷类别（25 类）
  natural_language_description     细粒度自然语言描述
  structured_attributes            结构化属性（Shape/Direction/Distribution/Scale/...）
  template_sentence                粗粒度模板句

用法:
  # 1) 冒烟测试：只取 1/10 数据，跑通全流程
  python build_vqa_dataset.py --subset 0.1

  # 2) 全量构造（含少数类过采样）
  python build_vqa_dataset.py --min-per-class 100

输出 (默认 data/ 目录下):
  steel_vqa_train.jsonl
  steel_vqa_val.jsonl
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download

REPO = "Zhaosxian/SteelDefectX"
ENDPOINT = "https://hf-mirror.com"  # 国内镜像
SPLITS = {"train": "train-text.json", "val": "val-text.json"}

# 25 类缺陷中文名（标准术语，可自行微调）
CLASS_CN = {
    "Bright scratch": "亮划痕",
    "Crazing": "龟裂",
    "Crease": "折痕",
    "Crescent gap": "月牙形缺口",
    "Dark scratches": "暗划痕",
    "Finishing roll printing": "精轧辊印",
    "Inclusion": "夹杂",
    "Iron scale compression": "氧化铁皮压入",
    "Iron sheet ash": "铁板灰",
    "Oil spot": "油斑",
    "Oxide scale of plate system": "板系氧化铁皮",
    "Oxide scale of temperature system": "温度系氧化铁皮",
    "Patches": "斑块",
    "Pitted surface": "麻点面",
    "Punching": "冲孔",
    "Red iron sheet": "红铁皮",
    "Rolled in scale": "轧入氧化皮",
    "Rolled pit": "轧制凹坑",
    "Secondary rust skin": "二次锈皮",
    "Silk spot": "丝斑",
    "Slag inclusion": "夹渣",
    "Waist folding": "腰部折叠",
    "Water spot": "水斑",
    "Welding line": "焊缝线",
    "White rust": "白锈",
}

# 3 个问题模板（对应检测/描述/判定三种能力）
QUESTIONS = [
    "请检测这张钢材表面图像中是否存在缺陷。如有，请以 JSON 格式输出缺陷类型、位置、严重程度和处理建议。",
    "请详细描述这张图像中可见的钢材表面缺陷特征，包括类型、形态和可能成因。",
    "这张钢材表面图像的质量是否合格？请给出判定依据和结构化检测报告。",
]


def derive_severity(saliency, scale):
    """由 Saliency + Scale 规则推导严重程度（规则可解释，非幻觉）"""
    score = 0
    score += {"high": 2, "medium": 1, "low": 0}.get(saliency, 0)
    score += {"extensive": 2, "large": 2, "medium": 1, "small": 1, "tiny": 0}.get(scale, 0)
    if score >= 3:
        return "严重"
    if score >= 1:
        return "中等"
    return "轻微"


def derive_suggestion(severity):
    return {
        "严重": "建议报废处理或立即停机检修",
        "中等": "建议返修或降级处理",
        "轻微": "建议观察跟踪或正常放行",
    }[severity]


def summarize_position(pos_list):
    """Position 形如 [['bottom-center',0.47],...]，取 Top3"""
    if not pos_list:
        return "unknown"
    top = sorted(pos_list, key=lambda x: -x[1])[:3]
    return ", ".join(f"{name}({prob:.2f})" for name, prob in top)


def build_answer(row):
    sa = row["structured_attributes"] or {}
    saliency = sa.get("Saliency", "")
    scale = sa.get("Scale", "")
    severity = derive_severity(saliency, scale)

    answer = {
        "defect_type": row["class_name"],
        "defect_type_cn": CLASS_CN.get(row["class_name"], row["class_name"]),
        "severity": severity,
        "suggestion": derive_suggestion(severity),
        "description": row.get("natural_language_description", ""),
        "attributes": {
            "shape": sa.get("Shape", ""),
            "direction": sa.get("Direction", ""),
            "spatial_distribution": sa.get("Spatial Distribution", ""),
            "number_of_defects": sa.get("Number of Defects", ""),
            "scale": sa.get("Scale", ""),
            "polarity": sa.get("Polarity", ""),
            "saliency": sa.get("Saliency", ""),
            "position": summarize_position(sa.get("Position", [])),
        },
    }
    return json.dumps(answer, ensure_ascii=False)


def download_one(args):
    """下载单个文件（显式 endpoint，绕开 list_repo_files 的镜像 bug）"""
    split, image_name, out_root, cache_dir = args
    repo_path = f"{split}/{image_name}"
    local_path = os.path.join(out_root, split, image_name)
    if os.path.exists(local_path):
        return local_path, True
    try:
        hf_hub_download(
            REPO, repo_path,
            repo_type="dataset",
            endpoint=ENDPOINT,
            cache_dir=cache_dir,
            local_dir=out_root,        # 直接落到本地目录结构
        )
        return local_path, os.path.exists(local_path)
    except Exception as e:
        return local_path, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=float, default=1.0, help="数据比例, 0.1 表示冒烟测试取 1/10")
    ap.add_argument("--min-per-class", type=int, default=0, help="少数类过采样下限, 0 表示不过采样")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__)))
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", ".hf_cache"))
    ap.add_argument("--workers", type=int, default=16, help="图片并行下载线程数")
    args = ap.parse_args()

    out_root = os.path.join(args.out_dir, "images")
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 下载 3 个文本 JSON
    for split, fn in SPLITS.items():
        local = os.path.join(args.out_dir, fn)
        if not os.path.exists(local):
            hf_hub_download(REPO, fn, repo_type="dataset", endpoint=ENDPOINT,
                            cache_dir=cache_dir, local_dir=args.out_dir)
    class_desc_path = os.path.join(args.out_dir, "class_descriptions.json")
    if not os.path.exists(class_desc_path):
        hf_hub_download(REPO, "class_descriptions.json", repo_type="dataset",
                        endpoint=ENDPOINT, cache_dir=cache_dir, local_dir=args.out_dir)

    # 2. 读取并做子集筛选（分层抽样）
    all_rows = []
    for split, fn in SPLITS.items():
        with open(os.path.join(args.out_dir, fn), encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            r["_split"] = split
        all_rows.extend(data)

    selected = []
    if args.subset < 1.0:
        # 按 (split, 类别) 分层抽样，保证 train/val 各自、每类都被覆盖
        by_key = defaultdict(list)
        for r in all_rows:
            by_key[(r["_split"], r["class_name"])].append(r)
        for (split, cls), rows in by_key.items():
            n = max(1, int(len(rows) * args.subset))
            selected.extend(rows[:n])
    else:
        selected = all_rows

    print(f"筛选后样本数: {len(selected)}")

    # 3. 并行下载所需图片（只下载被选中的）
    tasks = [(r["_split"], r["image_name"], out_root, cache_dir) for r in selected]
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, t) for t in tasks]
        for fut in as_completed(futs):
            _, succ = fut.result()
            ok += int(succ)
    print(f"图片下载: 成功 {ok}/{len(tasks)}")

    # 4. 少数类过采样（仅 train）
    if args.min_per_class > 0:
        train_rows = [r for r in selected if r["_split"] == "train"]
        cnt = Counter(r["class_name"] for r in train_rows)
        extra = []
        for cls, c in cnt.items():
            if c < args.min_per_class:
                pool = [r for r in train_rows if r["class_name"] == cls]
                need = args.min_per_class - c
                extra.extend(pool * (need // len(pool)) + pool[: need % len(pool)])
        selected = selected + extra
        print(f"过采样后样本数: {len(selected)} (+{len(extra)})")

    # 5. 构造 VQA 三元组并写 JSONL
    for split, fn in SPLITS.items():
        out_path = os.path.join(args.out_dir, f"steel_vqa_{split}.jsonl")
        split_rows = [r for r in selected if r["_split"] == split]
        with open(out_path, "w", encoding="utf-8") as f:
            for i, r in enumerate(split_rows):
                answer = build_answer(r)
                img_path = os.path.join("images", split, r["image_name"])  # 相对路径
                for j, q in enumerate(QUESTIONS):
                    sample = {
                        "id": f"{split}_{r['image_name'].replace('.jpg','')}_{j}",
                        "image": img_path,
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{q}"},
                            {"from": "gpt", "value": answer},
                        ],
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"已生成 {out_path}：{len(split_rows)} 张图 × {len(QUESTIONS)} 问 = {len(split_rows)*len(QUESTIONS)} 条")


if __name__ == "__main__":
    main()
