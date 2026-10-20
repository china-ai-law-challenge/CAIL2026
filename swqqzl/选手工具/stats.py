# -*- coding: utf-8 -*-
"""数据集统计脚本

用法：
    python stats.py "<jsonl路径>"        不传参数时默认统计 02_选手数据集/训练集_含真值/train.jsonl

输出：规模、内容形态/可见范围/场景/来源分布；若文件含真值，追加子任务 A/B/C 的标签分布。
"""
import json
import sys
import os
import collections

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.path.join(root, "02_选手数据集", "训练集_含真值", "train.jsonl"),
        os.path.join(root, "data", "train.jsonl"),
        # 主办方包布局（无 02_选手数据集，退化为统计标准答案）
        os.path.join(root, "03_标准答案", "prelim_key.jsonl"),
    ]
    path = next((p for p in cands if os.path.exists(p)), cands[0])
if not os.path.exists(path):
    sys.exit("文件不存在：%s\n（请显式传入一个 jsonl 路径，例如：python stats.py \"02_选手数据集/训练集_含真值/train.jsonl\"）" % path)

rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print("文件: %s" % path)
print("样本数: %d" % len(rows))
print("文本平均长度: %.1f 字符"
      % (sum(len(r["content"]["content_raw"]) for r in rows) / len(rows)))

for field, fn in [
    ("内容形态", lambda r: r["content"]["content_form"]),
    ("可见范围", lambda r: r["content"]["visibility"]),
    ("场景", lambda r: r["meta"]["domain_name"]),
    ("构造来源", lambda r: r["meta"]["source_category"]),
]:
    print()
    print("%s:" % field)
    for k, v in collections.Counter(fn(r) for r in rows).most_common():
        print("  %s: %d" % (k, v))

tr = [r for r in rows if r.get("truth")]
if not tr:
    print()
    print("（本文件不含真值，跳过子任务标签分布）")
    sys.exit(0)

print()
print("子任务A 侵权类型:")
for k, v in collections.Counter(
        t for r in tr for t in r["truth"]["task_A"]["infringement_types"]).most_common():
    print("  %s: %d" % (k, v))

for name, fn in [
    ("子任务A 边界结论", lambda t: [t["task_A"]["boundary_conclusion"]]),
    ("子任务A 严重程度", lambda t: [t["task_A"]["severity"]]),
    ("子任务B 发布者责任", lambda t: [t["task_B"]["publisher_liability"]]),
    ("子任务B 组织者责任", lambda t: [t["task_B"]["organizer_liability"]]),
    ("子任务B 平台义务", lambda t: t["task_B"]["platform_obligation"]),
    ("子任务B 平台责任", lambda t: [t["task_B"]["platform_liability"]]),
    ("子任务C 处置措施", lambda t: t["task_C"]["disposal_measures"]),
]:
    print()
    print("%s:" % name)
    for k, v in collections.Counter(
            x for r in tr for x in fn(r["truth"])).most_common():
        print("  %s: %d" % (k, v))
