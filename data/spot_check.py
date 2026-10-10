# -*- coding: utf-8 -*-
"""
VQA 数据人工抽查工具（阶段二 Step 3）

生成一个 HTML 报告，把随机抽到的样本的「图片 + 完整答案」并排显示，
方便你在浏览器里逐张核对：图里看到的缺陷 是否 = 答案里的 defect_type / 描述。

用法:
  python spot_check.py --n 30          # 抽查 30 张图
  python spot_check.py --n 30 --seed 7 # 换一批随机种子
  然后浏览器打开 spot_check.html
"""
import argparse
import base64
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="抽查图片数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="spot_check.html")
    args = ap.parse_args()

    data_dir = Path(__file__).parent

    # 按图去重（每张图在 JSONL 里是 3 行，只保留 1 条代表）
    by_img = {}
    for split in ["train", "val"]:
        fn = data_dir / f"steel_vqa_{split}.jsonl"
        if not fn.exists():
            continue
        for line in fn.read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            img_id = s["id"].rsplit("_", 1)[0]  # 去掉问题序号后缀
            by_img.setdefault(img_id, s)

    random.seed(args.seed)
    picks = random.sample(list(by_img.values()), min(args.n, len(by_img)))

    parts = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>VQA 数据抽查报告</title>
<style>
body{font-family:sans-serif;margin:24px;background:#fafafa}
h2{color:#333}
.item{display:flex;gap:18px;background:#fff;border:1px solid #e0e0e0;
      border-radius:10px;padding:14px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.item img{width:256px;height:256px;object-fit:contain;background:#111;border-radius:6px;flex:0 0 256px}
.meta{flex:1;min-width:0}
h4{margin:0 0 6px;font-size:15px}
.tag{display:inline-block;background:#eef4ff;color:#2456c4;padding:2px 8px;border-radius:12px;font-size:12px;margin-right:6px}
.q{color:#555;font-size:13px;margin:6px 0}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:12.5px;line-height:1.5}
.sev-严重{color:#d33;font-weight:bold}.sev-中等{color:#e90}.sev-轻微{color:#187}
</style></head><body>
<h2>VQA 数据抽查报告（%d 张图）</h2>
<p style="color:#666">核对方法：看左边图片里的真实缺陷，是否和右边答案里的 <b>defect_type_cn</b> 一致，且 <b>description/attributes</b> 的描述（形状、方向、位置）是否吻合。</p>
""" % len(picks)]

    for s in picks:
        img_path = data_dir / s["image"]
        b64 = base64.b64encode(img_path.read_bytes()).decode() if img_path.exists() else ""
        q = s["conversations"][0]["value"].replace("<image>\n", "")
        ans = json.loads(s["conversations"][1]["value"])
        sev = ans.get("severity", "")
        parts.append(f"""
<div class="item">
  <img src="data:image/jpeg;base64,{b64}" alt="{s['image']}"/>
  <div class="meta">
    <h4>{s['image']}
      <span class="tag">类型: {ans['defect_type_cn']} / {ans['defect_type']}</span>
      <span class="sev-{sev}">严重度: {sev}</span>
    </h4>
    <div class="q"><b>问题:</b> {q}</div>
    <pre>{json.dumps(ans, ensure_ascii=False, indent=2)}</pre>
  </div>
</div>""")

    parts.append("</body></html>")
    out = Path(args.out)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"✅ 已生成 {out}（共 {len(picks)} 张图）")
    print("   用浏览器打开，逐张核对图片与答案是否一致")


if __name__ == "__main__":
    main()
