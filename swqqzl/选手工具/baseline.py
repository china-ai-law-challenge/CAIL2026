# -*- coding: utf-8 -*-
"""下限基线：只用 train 的「全局众数 / 高频编码」生成格式合法的提交。

目的：提供一份**可直接运行、格式正确**的最小骨架与分数下限，而非性能上限。
纯标准库实现，无第三方依赖。

用法（发布包内路径，训练集与初赛集均已自动探测）：
    python baseline.py --out ../submission_baseline.jsonl
    python baseline.py --data "../02_选手数据集/初赛_无真值/prelim.jsonl" --out ../submission_baseline.jsonl
"""
import json
import os
import sys
import collections
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _first_existing(*rel):
    """按候选顺序取第一个存在的路径（发布包布局 `02_选手数据集/…` / 仓库布局 `data/…`）。"""
    for r in rel:
        p = os.path.join(ROOT, r)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, rel[0])


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _urgency(codes, prop_level, base_order, modifier):
    """基础紧迫度序 + 案情档位修正（与 labels.json.disposal_urgency_modifier 一致）。"""
    seq = sorted(codes, key=lambda m: base_order.index(m) if m in base_order else 999)
    if modifier and prop_level is not None and modifier.get("trigger") == prop_level:
        pro = [m for m in seq if m in (modifier.get("promote") or [])]
        rest = [m for m in seq if m not in (modifier.get("promote") or [])]
        return pro + rest
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=_first_existing(
        os.path.join("02_选手数据集", "训练集_含真值", "train.jsonl"),
        os.path.join("data", "train.jsonl")))
    ap.add_argument("--data", default=_first_existing(
        os.path.join("02_选手数据集", "初赛_无真值", "prelim.jsonl"),
        os.path.join("data", "prelim.jsonl")))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission_baseline.jsonl"))
    a = ap.parse_args()

    lab = json.load(open(os.path.join(ROOT, "schema", "labels.json"), encoding="utf-8"))
    ELEMS = list(lab["elements"])
    NEC = set(lab["element_necessity"]["necessary"])
    URG = list(lab["disposal_urgency_order"])
    UMOD = lab.get("disposal_urgency_modifier") or {}
    LAW = {t["code"]: t.get("law", []) for t in lab["infringement_types"]}
    ANC = lab["type_case_anchors"]
    TNAME = {t["code"]: t["name"] for t in lab["infringement_types"]}

    for _p, _label, _flag in ((a.train, "训练集", "--train"), (a.data, "输入数据", "--data")):
        if not os.path.exists(_p):
            sys.exit(
                "找不到%s：%s\n"
                "  本脚本需要「含真值的训练集」。请确认你解压的是**完整包**，或用 %s 显式指定路径，例如：\n"
                "    python baseline.py %s \"<路径>\" --out submission_baseline.jsonl"
                % (_label, _p, _flag, _flag))
    tr = load(a.train)
    rows = load(a.data)

    def maj(fn):
        return collections.Counter(fn(r) for r in tr).most_common(1)[0][0]

    def topk(fn, k):
        c = collections.Counter(x for r in tr for x in fn(r))
        return [x for x, _ in c.most_common(k)]

    bnd = maj(lambda r: r["truth"]["task_A"]["boundary_conclusion"])
    sev = maj(lambda r: r["truth"]["task_A"]["severity"])
    pub = maj(lambda r: r["truth"]["task_B"]["publisher_liability"])
    org = maj(lambda r: r["truth"]["task_B"]["organizer_liability"])
    plt = maj(lambda r: r["truth"]["task_B"]["platform_liability"])
    og = maj(lambda r: bool(r["truth"]["task_B"]["organized_infringement"]))
    obl = topk(lambda r: r["truth"]["task_B"]["platform_obligation"], 1)
    ful = maj(lambda r: bool(r["truth"]["task_B"]["platform_obligation_fulfilled"]))
    types = topk(lambda r: r["truth"]["task_A"]["infringement_types"], 2)
    meas = topk(lambda r: r["truth"]["task_C"]["disposal_measures"], 3)

    ELEM_OF = {i["code"]: i["elements"] for i in lab["infringement_types"]}
    in_scope = {x for t in types for x in ELEM_OF.get(t, [])}
    ej = {e: (("构成" if e in NEC else "存疑") if e in in_scope else "不适用") for e in ELEMS}
    law = sorted({c for t in types for c in LAW.get(t, [])})
    cites = law + sorted({x for t in types for x in ANC.get(t, [])})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            # 边界样本：措施类字段全空（「不得过度处置」判定点）
            empty = bnd in ("不构成侵权", "存疑待查")
            pred = {
                "task_A": {"infringement_types": [] if empty else types, "boundary_conclusion": bnd,
                           "exemptions": [], "element_judgments": ej, "severity": sev},
                "task_B": {"publisher_liability": pub, "organizer_liability": org,
                           "platform_obligation": obl, "platform_obligation_fulfilled": ful,
                           "platform_liability": plt, "organized_infringement": og},
                "task_C": {"disposal_measures": [] if empty else meas,
                           "priority_order": [] if empty else _urgency(
                               meas, r["context"]["propagation_level"], URG, UMOD),
                                                      },
                "citations": [] if empty else cites,
            }
            f.write(json.dumps({"sample_id": r["sample_id"], "prediction": pred},
                               ensure_ascii=False) + "\n")
    print("已写出 %s（%d 条）——格式合法的下限基线，仅供对照" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
