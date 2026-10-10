# -*- coding: utf-8 -*-
"""
Qwen2.5-VL LoRA 微调脚本（阶段三，标准 transformers + peft 方案）

不用 unsloth / trl，直接基于 transformers Trainer + peft + 自定义数据整理，
依赖少、可控性强，底层就是主流 LoRA。

用法:
  # 冒烟测试：只取 100 条，跑 10 步，确认能训练
  python train_lora.py --max-samples 100 --max-steps 10

  # 正式训练：全量数据，1 个 epoch
  python train_lora.py --epochs 1

环境要求（steel-vqa 环境）:
  torch 2.5.1+cu124 / transformers>=4.49 / peft / datasets / accelerate
"""
import argparse
import json
import os
import random

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import swanlab

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_samples(data_dir, jsonl_name, max_samples):
    """读阶段二 JSONL，转成 [{'messages': [...], 'image_path': ...}]"""
    samples = []
    with open(os.path.join(data_dir, jsonl_name), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            question = s["conversations"][0]["value"].replace("<image>\n", "")
            answer = s["conversations"][1]["value"]
            samples.append({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": answer},
                ],
                "image_path": os.path.join(data_dir, s["image"]),
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples


class VLMDataset(Dataset):
    """预处理成模型可直接吃的 tensor（labels 只监督 assistant 回答部分）"""

    def __init__(self, samples, processor):
        self.processor = processor
        self.processed = [self._preprocess(s) for s in samples]

    @staticmethod
    def _augment(image):
        """数据增强：亮度/对比度抖动。

        只做光照类增强，不做翻转/旋转——因为答案含"位置/方向"字段，
        翻转会破坏这些标签（如 bottom-right 变成 bottom-left）。
        """
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return image

    def _preprocess(self, s):
        image = Image.open(s["image_path"]).convert("RGB")
        image = self._augment(image)  # 数据增强
        # 完整对话（问题+答案），以及仅问题（带 assistant 开头），用于掩码 labels
        full_text = self.processor.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(
            [s["messages"][0]], tokenize=False, add_generation_prompt=True)

        full = self.processor(text=[full_text], images=[image], return_tensors="pt")
        prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt")

        labels = full["input_ids"][0].clone()
        labels[: prompt["input_ids"].shape[1]] = -100  # 掩掉问题部分

        return {
            "input_ids": full["input_ids"][0],
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "pixel_values": full["pixel_values"],        # [num_patches, 1176]，不要 [0] 切片
            "image_grid_thw": full["image_grid_thw"][0],  # [3] = [t, h, w]
        }

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, i):
        return self.processed[i]


def collate_fn(batch, pad_token_id):
    max_len = max(b["input_ids"].shape[0] for b in batch)

    def pad(t, val):
        return torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=val)

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_token_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        # pixel_values 需按 patch 拼接成 [总patch数, 1176]，image_grid_thw 堆叠成 [batch, 3]
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--max-samples", type=int, default=0, help=">0 时只取前 N 条")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 覆盖 epochs")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "outputs", "steel-vqa-lora"))
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # 内置，无需额外装 flash-attn
    )

    # 无需手动冻结视觉编码器：get_peft_model 会自动冻结所有非 LoRA 参数。
    # 由于 target_modules 只命中语言模型的 q_proj/k_proj/... （视觉塔用 qkv/proj 命名，不冲突），
    # 视觉编码器自然被冻结，只有语言侧的 LoRA 适配器参与训练。

    # LoRA 只挂在语言模型的注意力/MLP 上（视觉塔的 qkv 命名不同，不会被误伤）
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = load_samples(args.data_dir, "steel_vqa_train.jsonl", args.max_samples)
    print(f"训练样本数: {len(samples)}")
    dataset = VLMDataset(samples, processor)

    # 计算 warmup 步数（transformers 5.x 已移除 warmup_ratio，只能用 warmup_steps）
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = int(len(dataset) / (args.batch_size * args.grad_accum) * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.03))

    # 实验跟踪（本地模式，无需注册，浏览器打开终端打印的网址即可看 loss 曲线）
    swanlab.init(
        project="steel-vqa",
        config={"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum},
        mode="local",
    )

    pad_token_id = processor.tokenizer.pad_token_id
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            seed=3407,
            remove_unused_columns=False,
            report_to="swanlab",
        ),
        train_dataset=dataset,
        data_collator=lambda b: collate_fn(b, pad_token_id),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA 适配器已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
