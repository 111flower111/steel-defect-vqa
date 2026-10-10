# -*- coding: utf-8 -*-
"""
修补 vLLM 0.7.2 的 Qwen2.5-VL 类名 bug

问题：vLLM 0.7.2 的 qwen2_5_vl.py 引用了不存在的 Qwen2_5_VLImageProcessor
      （正确类名是 Qwen2VLImageProcessor，位于 transformers.models.qwen2_vl）

用法（重装 vLLM 后需重新执行）:
  conda activate vllm-deploy
  python patch_vllm.py
"""
import vllm
from pathlib import Path

path = Path(vllm.__file__).parent / "model_executor/models/qwen2_5_vl.py"
code = path.read_text()

old_import = ("from transformers.models.qwen2_5_vl import (Qwen2_5_VLImageProcessor,\n"
              "                                            Qwen2_5_VLProcessor)")
new_import = ("from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor\n"
              "from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor")

if "Qwen2_5_VLImageProcessor" not in code:
    print("无需修补（已修复或类名不同）")
else:
    code = code.replace(old_import, new_import)
    code = code.replace("Qwen2_5_VLImageProcessor", "Qwen2VLImageProcessor")
    path.write_text(code)
    print(f"✅ 已修补 {path}")
