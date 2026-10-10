# 部署说明（vLLM 版）

## 架构

```
浏览器(http://127.0.0.1:8001) → FastAPI 封装(main_vllm.py) → vLLM(端口8000) → 合并模型
```

## 一键启动 / 停止

```bash
cd /home/wxy/factory_test/steel-defect-vqa/deploy
bash start.sh    # 启动（约 30~60 秒加载模型）
bash stop.sh     # 停止
```

启动后浏览器打开 **http://127.0.0.1:8001**，上传图片即可生成检测报告。

## 环境说明（两个独立 conda 环境）

| 环境 | 用途 | 关键包 |
|---|---|---|
| `vllm-deploy` (python 3.11) | vLLM 推理引擎 | vllm 0.7.2 + torch 2.5.1+cu124 + transformers 4.50 |
| `steel-vqa` (python 3.12) | FastAPI 封装 | fastapi + httpx |

> ⚠️ 两个环境必须分开：vLLM 对 torch/transformers 版本有硬性要求，与训练环境冲突。

## 关键坑与修复（务必记录）

1. **vLLM 版本**：最新 vLLM（0.9+）要求 CUDA≥12.8，本机驱动 535（CUDA 12.2）装不上 → 用 **vLLM 0.7.2**（CUDA 12.4）。

2. **transformers 版本**：vLLM 0.7.2 默认拉取 transformers 5.x（需 torch 2.6+），报 `torch.accelerator` 错误 → 降到 **transformers 4.50**。

3. **类名 bug**：vLLM 0.7.2 的 `qwen2_5_vl.py` 引用了不存在的 `Qwen2_5_VLImageProcessor`（正确名是 `Qwen2VLImageProcessor`）→ 已手动修补：
   ```python
   # vllm/model_executor/models/qwen2_5_vl.py
   from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
   ```
   修补方法见 `patch_vllm.py`。

## 模型产物路径

| 文件 | 路径 |
|---|---|
| LoRA 适配器（增强版 94.0%） | `train/outputs/steel-vqa-lora-aug` |
| 合并后完整模型 | `train/outputs/steel-vqa-merged` |
