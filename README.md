# 基于 Qwen2.5-VL 的钢材表面缺陷多模态结构化报告生成系统

面向工业质检场景，基于 **SteelDefectX** 多模态数据集，构建「钢材表面图片 → 结构化缺陷报告」的端到端多模态大模型系统。覆盖 **数据构建 → LoRA 微调 → 数据增强 → AWQ INT4 量化 → vLLM 部署** 全链路。

---

## 📊 核心成果

| 指标 | 结果 |
|---|---|
| 缺陷分类准确率 | **94.0%**（数据增强后） |
| JSON 格式合规率 | **100%** |
| 模型体积压缩 | **7.1GB → 3.2GB（2.2 倍）** |
| 训练可训练参数 | 0.98%（37M / 3.79B） |
| 端到端部署 | vLLM + FastAPI + 网页，图片上传 → 报告生成 |

---

## 🏗️ 技术架构

```
┌─────────────┐    ┌────────────┐    ┌──────────────┐    ┌─────────────────┐
│  网页前端    │───→│  FastAPI   │───→│     vLLM     │───→│ Qwen2.5-VL-3B   │
│ (图片上传)   │    │  (接口)    │    │ (awq_marlin) │    │  AWQ INT4 量化   │
│ (报告展示)   │←───│            │←───│              │←───│                 │
└─────────────┘    └────────────┘    └──────────────┘    └─────────────────┘
```

**技术栈**：PyTorch · Transformers · PEFT(LoRA) · Qwen2.5-VL · AutoAWQ · vLLM · FastAPI · SwanLab

---

## 📁 项目结构

```
steel-defect-vqa/
├── data/
│   ├── explore_data.py          # 数据集探索（字段/类别分布）
│   ├── build_vqa_dataset.py     # VQA 指令数据构造
│   └── DATA_FINDINGS.md         # 数据探索结论
├── train/
│   ├── train_lora.py            # LoRA 微调（含数据增强）
│   ├── merge_lora.py            # LoRA 权重合并
│   └── test_inference.py        # 推理验证
├── quantize/
│   └── awq_quantize.py          # AWQ INT4 量化
├── eval/
│   ├── evaluate.py              # bf16 模型评估
│   └── evaluate_awq.py          # 量化模型评估（vLLM API）
├── deploy/
│   ├── main_vllm.py             # FastAPI 封装（对接 vLLM）
│   ├── index.html               # 网页前端
│   ├── start.sh / stop.sh       # 一键启停
│   └── README.md                # 部署文档
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 训练环境
conda create -n steel-vqa python=3.12 -y
conda activate steel-vqa
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.51.0" "peft==0.14.0" datasets pillow pandas tqdm swanlab

# 部署环境（vLLM 需独立环境，版本冲突）
conda create -n vllm-deploy python=3.11 -y
conda activate vllm-deploy
pip install "vllm==0.7.2" "transformers==4.50.0"

# 量化环境
conda create -n quantize2 python=3.11 -y
conda activate quantize2
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "autoawq==0.2.9" "transformers==4.51.3" "datasets<4"
```

> ⚠️ 国内网络需设置 `export HF_ENDPOINT=https://hf-mirror.com` 和 `export HF_HUB_DISABLE_XET=1`

### 2. 数据构造

```bash
cd data
python explore_data.py                        # 探索数据集
python build_vqa_dataset.py --min-per-class 100   # 构造 VQA 数据
```

### 3. LoRA 微调

```bash
cd train
python train_lora.py --epochs 1               # 训练（含亮度/对比度增强）
python merge_lora.py                          # 合并 LoRA
```

### 4. 评估

```bash
cd eval
python evaluate.py --max-samples 400          # bf16 模型评估
```

### 5. AWQ INT4 量化

```bash
conda activate quantize2
cd quantize
python awq_quantize.py                        # 量化（7.1GB → 3.2GB）
```

### 6. 部署

```bash
cd deploy
bash start.sh                                 # 启动 vLLM + FastAPI
# 浏览器打开 http://127.0.0.1:8001
```

---

## 🔬 技术亮点

### 数据构建
- 解析 SteelDefectX 的粗/细粒度标注，设计 3 类问题模板（检测/描述/判定）
- 构建 **23,877 条** VQA 问答对，结构化 JSON 答案（缺陷类型/位置/严重度/建议）
- 针对 **15.9 倍**类别不均衡做少数类过采样

### 微调策略
- **LoRA r=16**，冻结视觉编码器、仅训练语言侧（可训练参数 0.98%）
- 数据增强：**仅亮度/对比度抖动**（翻转/旋转会破坏答案中的"位置/方向"标签一致性）
- 准确率：**92.8% → 94.0%**

### 量化
- **AWQ INT4**（group_size=128），自动排除视觉塔（`modules_to_not_convert=["visual"]`）
- 模型体积 **7.1GB → 3.2GB**，JSON 合规率保持 100%

### 部署
- **vLLM**（awq_marlin 内核）+ **FastAPI** + 网页前端
- OpenAI 兼容 API，端到端延迟低

---

## ⚠️ 踩坑记录（工程经验）

| 问题 | 原因 | 解决 |
|---|---|---|
| `load_dataset` 无法访问 | 镜像站不支持 `HfFileSystem` 的 endpoint 覆盖 | 用 `hf_hub_download(repo_type="dataset")` 逐文件下载 |
| vLLM 加载 Qwen2.5-VL 报类名错误 | vLLM 0.7.2 引用了不存在的 `Qwen2_5_VLImageProcessor` | 修补为 `Qwen2VLImageProcessor`（见 `patch_vllm.py`） |
| transformers 5.x 报 `torch.accelerator` | 新版 transformers 需 torch 2.6+ | 降级到 transformers 4.50/4.51 |
| FP8 量化报 `size_n not divisible by 64` | A800 无原生 FP8，退回 Marlin 内核且维度不整除 | 改用 AWQ INT4 |
| AWQ 量化多卡设备不匹配 | `device_map="auto"` 跨卡 | 改用单卡 `device_map="cuda:0"` |
| 校准数据集下载慢 | 默认 pileval 需从 HF 下载 | 改用项目自身问答文本做校准 |

---

## 📄 License

本项目基于 Qwen2.5-VL（Apache 2.0）与 SteelDefectX 数据集构建，仅供学习研究使用。
