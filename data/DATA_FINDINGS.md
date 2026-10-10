# SteelDefectX 数据集探索结论（阶段一）

## 1. 环境

| 项 | 状态 |
|---|---|
| Python | 3.12.7（base anaconda） |
| 虚拟环境 | `/home/wxy/factory_test/.venv`（venv，因文件沙箱限制未用 conda） |
| 关键包 | datasets 5.0.1 / pandas 3.0.6 / pillow 12.3.0 / tqdm 4.70.1 / huggingface_hub 1.33.0 |
| GPU | ✅ 2× NVIDIA A800 80GB（驱动 535.183.01，CUDA 12.2；base 环境 torch 2.5.1+cu124，`cuda_available=True`） |
| 网络 | ✅ hf-mirror.com 可用；pip 用清华镜像 |

### 环境搭建命令（已执行）
```bash
cd /home/wxy/factory_test
python3 -m venv .venv
TMPDIR=$PWD/.tmp .venv/bin/pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  datasets pillow pandas tqdm
```
> 注意：pip 默认缓存目录在 `~/.cache/pip`，在受限环境下会卡住，需 `--no-cache-dir`。

## 2. 数据集真实结构（与方案假设不一致）

- 数据集 **非 gated、公开**，无需登录。
- `format: imagefolder` —— 图片与文本标注 **分开存放**，`load_dataset` 不会自动关联文本。
- 元数据：`size_categories:10K<n<100K`，`license:cc-by-4.0`。

### 实际文件清单
```
train/           5454 张图（256×256 RGB JPEG，单张 4~15KB）
val/             2324 张图
train_mask/      掩码（本 VQA 项目暂不需要）
val_mask/        掩码
train-text.json  5454 条文本标注（7.6MB）
val-text.json    2324 条文本标注（3.4MB）
class_descriptions.json  25 类缺陷的英文描述
```

### 文本标注字段（每条一个 dict）
| 字段 | 含义 | 示例 |
|---|---|---|
| `image_name` | 图片文件名（对应 `train/` 或 `val/` 下同名文件） | `bs_01.jpg` |
| `class_name` | 缺陷类别（25 类之一） | `Bright scratch` |
| `natural_language_description` | 细粒度自然语言描述 | "The bright scratch defect appears as a thin, vertical line..." |
| `structured_attributes` | 结构化属性字典（见下） | `{Defect type, Shape, Direction, ...}` |
| `template_sentence` | 粗粒度模板句 | "A tiny bright Bright scratch is observed..." |

> 方案里假设的 `label` / `coarse_text` / `fine_text` 字段名 **不存在**，实际是 `class_name` / `template_sentence` / `natural_language_description`。

### structured_attributes 取值空间（用于阶段二构造答案）
| 字段 | 取值 |
|---|---|
| Defect type | 25 类缺陷名 |
| Shape | irregular(2185) / linear(2121) / elongated(1674) / spot-like(744) / circular(370) / curved(318) / patch-like(244) / network-like(75) / diffuse(38) / unclear(6) / fragmented(3) |
| Direction | none(3670) / vertical(2986) / horizontal(729) / diagonal(393) |
| Spatial Distribution | scattered(3708) / isolated(2977) / clustered(1086) / unclear(7) |
| Number of Defects | one / two / three / four / five / multiple |
| Scale | medium / small / large / extensive / tiny |
| Polarity | dark(6188) / bright(827) / neutral(763) |
| Saliency | medium(3907) / high(1959) / low(1912) |
| Position | `[["bottom-center", 0.467], ...]`（位置名 + 置信度） |

## 3. 类别分布（25 类，train+val = 7778）

| 类别 | 总数 | train | val |
|---|---|---|---|
| Inclusion | 795 | 557 | 238 |
| Water spot | 700 | 490 | 210 |
| Silk spot | 578 | 405 | 173 |
| Welding line | 500 | 350 | 150 |
| Oil spot | 498 | 349 | 149 |
| Bright scratch | 429 | 301 | 128 |
| Punching | 399 | 280 | 119 |
| Red iron sheet | 397 | 278 | 119 |
| Dark scratches | 334 | 234 | 100 |
| Crazing | 300 | 210 | 90 |
| Patches | 300 | 210 | 90 |
| Pitted surface | 300 | 210 | 90 |
| Rolled in scale | 300 | 210 | 90 |
| Crescent gap | 245 | 172 | 73 |
| Slag inclusion | 238 | 167 | 71 |
| Finishing roll printing | 203 | 143 | 60 |
| Oxide scale of temperature system | 203 | 143 | 60 |
| Iron scale compression | 200 | 140 | 60 |
| Secondary rust skin | 198 | 139 | 59 |
| Waist folding | 198 | 139 | 59 |
| White rust | 153 | 108 | 45 |
| Iron sheet ash | 122 | 86 | 36 |
| Crease | 75 | 53 | 22 |
| Oxide scale of plate system | 63 | 45 | 18 |
| Rolled pit | 50 | 35 | 15 |

- **不均衡比 15.9x**（最多 Inclusion 795，最少 Rolled pit 50）。
- 少数类（<100）：Rolled pit(50) / Oxide scale of plate system(63) / Crease(75)，建议过采样。

## 4. image_name 前缀 → 类别映射（文件名命名规则）

| 前缀 | 类别 | 前缀 | 类别 |
|---|---|---|---|
| bs | Bright scratch | Isa | Iron sheet ash |
| cracking | Crazing | Isc | Iron scale compression |
| crease | Crease | Ops | Oxide scale of plate system |
| cg | Crescent gap | Ots | Oxide scale of temperature system |
| ds | Dark scratches | ris | Red iron sheet |
| frp | Finishing roll printing | rs | Rolled in scale |
| in | Inclusion | rp | Rolled pit |
| os | Oil spot | Srs | Secondary rust skin |
| pa | Patches | ss | Silk spot |
| ps | Pitted surface | si | Slag inclusion |
| pu | Punching | wf | Waist folding |
| wl | Welding line | wr | White rust |
| ws | Water spot | | |

## 5. 下载方案（重要）

`load_dataset("Zhaosxian/SteelDefectX")` 在本环境 **不可用**：
- imagefolder 构建器内部用 `HfFileSystem` 列文件，该组件**忽略 `HF_ENDPOINT` 镜像配置**，仍访问 `huggingface.co`，导致 DNS 解析失败。
- 即使能用，imagefolder 也只返回图片，**不会关联文本标注**。

**推荐下载方式**（已验证可用）：
```python
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    "Zhaosxian/SteelDefectX",
    "train/bs_01.jpg",
    repo_type="dataset",            # 必须指定，否则拼 URL 缺 /datasets/ 段
    endpoint="https://hf-mirror.com",
    cache_dir="/home/wxy/factory_test/.hf_cache",
)
```
或用 curl（已验证）：
```bash
curl -sL "https://hf-mirror.com/datasets/Zhaosxian/SteelDefectX/resolve/main/train/bs_01.jpg" -o out.jpg
```

## 6. 对后续阶段的影响

1. **阶段二数据构造**：需手动把 `image_name` 与图片路径 join，再用 `class_name` / `template_sentence` / `natural_language_description` / `structured_attributes` 构造「图像+问题+结构化答案」三元组。方案中的 `label/coarse_text/fine_text` 字段名需全部替换。
2. **少数类过采样**：Rolled pit / Oxide scale of plate system / Crease 三类样本 <100，需轻微过采样。
3. **阶段三训练**：本机 2× A800 80GB，算力充足，Qwen2.5-VL-3B 的 LoRA 微调可正常进行（3B 模型 4bit 量化后训练显存需求远低于 80GB）。
