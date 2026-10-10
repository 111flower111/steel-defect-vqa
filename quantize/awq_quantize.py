# -*- coding: utf-8 -*-
"""
AWQ INT4 量化脚本（Qwen2.5-VL-3B，阶段四）

用 AutoAWQ 0.2.9 把合并后的 bf16 模型量化为 AWQ INT4。
AutoAWQ 的 Qwen2.5-VL 实现会自动排除视觉塔（modules_to_not_convert=["visual"]），
只量化语言模型，与训练策略一致。

用法:
  conda activate quantize2
  HF_ENDPOINT=https://hf-mirror.com HF_HOME=... HF_HUB_DISABLE_XET=1 python awq_quantize.py
"""
import json
import os
import time

from awq import AutoAWQForCausalLM
from transformers import AutoProcessor


def build_calib_texts(data_dir, n=256):
    """用我们自己的钢材缺陷问答文本做校准（比 pileval 通用文本更贴合任务）"""
    texts = []
    with open(os.path.join(data_dir, "steel_vqa_train.jsonl"), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            q = s["conversations"][0]["value"].replace("<image>\n", "")
            a = s["conversations"][1]["value"]
            texts.append(f"{q}\n{a}")
            if len(texts) >= n:
                break
    return texts

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "train", "outputs", "steel-vqa-merged")
QUANT_PATH = os.path.join(BASE, "quantize", "awq-int4")

QUANT_CONFIG = {
    "zero_point": True,     # 零点量化
    "q_group_size": 128,    # 分组大小
    "w_bit": 4,             # 4bit
    "version": "GEMM",      # GEMM 内核
}


def main():
    t0 = time.time()
    print(f"[1/4] 加载模型: {MODEL_PATH} (单卡 cuda:0)")
    model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"      加载完成，耗时 {time.time()-t0:.0f}s")

    print("[2/4] 构造校准数据（用本项目钢材缺陷问答文本，256 条）...")
    calib_texts = build_calib_texts(os.path.join(BASE, "data"), n=256)
    print(f"      校准样本数: {len(calib_texts)}")

    print("[3/4] 开始 AWQ 量化 ...")
    model.quantize(processor.tokenizer, quant_config=QUANT_CONFIG, calib_data=calib_texts)
    print(f"      量化完成，耗时 {time.time()-t0:.0f}s")

    print(f"[4/4] 保存量化模型到: {QUANT_PATH}")
    os.makedirs(QUANT_PATH, exist_ok=True)
    model.save_quantized(QUANT_PATH)
    processor.save_pretrained(QUANT_PATH)

    print(f"✅ 量化完成，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
