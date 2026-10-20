# -*- coding: utf-8 -*-
"""官方参考解（中等强度基线）：仅用输入侧结构化字段的纯 Python 随机森林。

定位：
  · **不是**性能上限，而是「认真做就能达到」的参考线，用于判断提交是否跑通、
    以及对照各子任务的参考分区间（见 `01_赛道说明文档/06_选手使用与提交指南.md` 第九节）。
  · **不使用**任何答案键、**不读正文文本**、不使用初赛/复赛/封测的任何信息 ——
    全部由输入侧 25 个结构化字段从**训练集**学习，因此它的分数就是
    「结构化字段能走多远」的实测锚点。
  · 纯标准库（无 numpy / sklearn / torch），固定随机种子，结果可复现。
  · 本脚本**不是标准答案**，官方评分程序不向选手分发；它的用途是让你跑通流程、
    拿到一个分数下限，并对照赛题说明文档中的参考分区间校准预期。

实测（本版语料 / valid 1400 条，默认参数，约 29 秒）：
    A 边界 0.774｜A 类型 macro-F1 0.307｜要件 micro/macro 0.635/0.300｜严重程度 0.570
    B 责任主体 0.815｜平台义务 0.725｜义务履行状态 0.842｜平台责任 0.859｜组织化 0.970
    C 措施 0.767｜优先级 τ 0.722｜**数据溯源率 0.554（覆盖率 1.000 × 引用有效率 0.554）**
  注：溯源率仍未过 0.6 门限，说明**收紧口径后的门限项确实构成约束** —— 只有把「类型认定」做扎实
  （边界不误判、小众类型不漏）并把引用逐样本对齐，才能过线；靠铺满法条（valid 实测 0.183）与堆砌引用无效。
  本解的引用有效率只有 0.554（精确率 0.509 / 召回率 0.609），即引用给得多但不够准；
  这正是留给选手的改进空间。覆盖率已做到 1.000（应给引用的样本一条不漏）。
  注：数值随语料版本浮动，重跑 `reference.py --data <验证集路径>` 即可复现当前值。

用法（发布包内路径，训练集与初赛集均已自动探测，无需手工指定）：
    python reference.py --out submission_reference.jsonl
    python check_submission.py --pred submission_reference.jsonl \
           --data "../02_选手数据集/初赛_无真值/prelim.jsonl"
    python reference.py --data "../02_选手数据集/验证集_含真值/valid.jsonl" --out ref_valid.jsonl

★ 本脚本**不是标准答案**：官方评分程序由主办方统一执行，不随选手包分发。产出提交后请用
  `check_submission.py` 自检格式；选手侧若要自评，请按赛题说明文档给出的口径自行实现评测。

实现要点：
  · 单值任务（边界结论 / 严重程度 / 平台责任）→ 若干棵 CART 树投票；
  · 多标签任务（侵权类型 / 平台义务 / 处置措施）→ 每个标签独立建树做二分类投票；
  · 处置优先级按 labels.json 的「基础紧迫度序 + 案情档位修正」派生；
  · citations 由「预测类型 → 法条编码 / 案例锚点」映射生成（门限项「数据溯源率」）。
"""
import json
import os
import sys
import argparse
import collections
import random
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _first_existing(*rel):
    """按候选顺序取第一个存在的路径。

    发布包内数据位于 `02_选手数据集/…`，而开发态脚本默认写的是 `data/…`。
    这里同时探测两套位置，使脚本在「发布包」与「数据集仓库」两种布局下
    开箱即用，选手无需手工拼路径。
    """
    for r in rel:
        p = os.path.join(ROOT, r)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, rel[0])


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def feats(r):
    """抽取 25 个输入侧结构化字段（不含正文、不含任何真值）。"""
    c, x, s = r["content"], r["context"], r["subject"]
    return (
        c["content_form"], c["platform"], c["visibility"],
        int(bool(c["contains_personal_info"])), int(bool(c["involves_minor"])),
        "|".join(sorted(c["topic_tags"])),
        x["publisher_type"], int(bool(x["account_verified"])), x["follower_level"],
        x["history_behavior"], x["propagation_level"], int(bool(x["cross_platform"])),
        int(bool(x["hot_search"])), x["victim_response"], x["organized_clue"],
        x["content_truthfulness"], int(bool(x["publisher_verified"])),
        s["target_identity"], int(bool(s["platform_notified"])), s["platform_action"],
        s["delay_bucket"], s["platform_measures"], s["platform_complaint_channel"],
        int(bool(s["profit_clue"])), int(bool(s["consent_obtained"])),
    )


NF = 25


def gini(cnt):
    n = sum(cnt.values())
    return 1.0 - sum((v / n) ** 2 for v in cnt.values()) if n else 0.0


class Node:
    __slots__ = ("f", "val", "ch", "leaf", "p")

    def __init__(self):
        self.f = None
        self.val = None
        self.ch = None
        self.leaf = True
        self.p = 0.0


def build(X, y, idx, rng, maxd, minleaf, mtry, depth=0):
    node = Node()
    cnt = collections.Counter(y[i] for i in idx)
    node.val = cnt.most_common(1)[0][0]
    node.p = cnt.get(1, 0) / len(idx)
    if depth >= maxd or len(cnt) == 1 or len(idx) < 2 * minleaf:
        return node
    base = gini(cnt)
    best, bestgain = None, 0.0
    for f in rng.sample(range(NF), min(mtry, NF)):
        groups = collections.defaultdict(list)
        for i in idx:
            groups[X[i][f]].append(i)
        if len(groups) < 2:
            continue
        score = 0.0
        for ids in groups.values():
            score += len(ids) / len(idx) * gini(collections.Counter(y[i] for i in ids))
        gain = base - score
        if gain > bestgain + 1e-12:
            bestgain, best = gain, f
    if best is None:
        return node
    node.f, node.leaf, node.ch = best, False, {}
    groups = collections.defaultdict(list)
    for i in idx:
        groups[X[i][best]].append(i)
    for v, ids in groups.items():
        node.ch[v] = build(X, y, ids, rng, maxd, minleaf, mtry, depth + 1)
    return node


def _walk(node, x):
    while not node.leaf:
        nxt = node.ch.get(x[node.f])
        if nxt is None:
            break
        node = nxt
    return node


def forest(X, y, ntree, rng, maxd, minleaf, mtry):
    n = len(y)
    return [build(X, y, [rng.randrange(n) for _ in range(n)], rng, maxd, minleaf, mtry)
            for _ in range(ntree)]


def vote_multi(trees, x, thr=0.5):
    return sum(_walk(t, x).p for t in trees) / len(trees) > thr


def vote_single(trees, x):
    return collections.Counter(_walk(t, x).val for t in trees).most_common(1)[0][0]


def _urgency(codes, prop, base, mod):
    seq = sorted(set(codes), key=lambda m: base.index(m) if m in base else 999)
    if mod and mod.get("trigger") == prop:
        return ([m for m in seq if m in mod.get("promote", [])] +
                [m for m in seq if m not in mod.get("promote", [])])
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=_first_existing(
        os.path.join("02_选手数据集", "训练集_含真值", "train.jsonl"),
        os.path.join("data", "train.jsonl")))
    ap.add_argument("--data", default=_first_existing(
        os.path.join("02_选手数据集", "初赛_无真值", "prelim.jsonl"),
        os.path.join("data", "prelim.jsonl")))
    ap.add_argument("--out", default=os.path.join(ROOT, "submission_reference.jsonl"))
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--ntree-single", type=int, default=20)
    ap.add_argument("--ntree-multi", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--min-leaf", type=int, default=4)
    ap.add_argument("--mtry", type=int, default=6)
    ap.add_argument("--multi-threshold", type=float, default=0.30,
                    help="多标签判定的投票阈值：调低→召回提高（类型更全、溯源覆盖率更高），"
                         "调高→精确率提高")
    a = ap.parse_args()
    THR = a.multi_threshold

    t0 = time.time()
    lab = json.load(open(os.path.join(ROOT, "schema", "labels.json"), encoding="utf-8"))
    ELEMS = list(lab["elements"])
    NEC = set(lab["element_necessity"]["necessary"])
    ELEM_OF = {i["code"]: i["elements"] for i in lab["infringement_types"]}
    TNAME = {t["code"]: t["name"] for t in lab["infringement_types"]}
    LAW_OF = {t["code"]: t.get("law", []) for t in lab["infringement_types"]}
    ANC_OF = lab["type_case_anchors"]
    URG = list(lab["disposal_urgency_order"])
    UMOD = lab.get("disposal_urgency_modifier") or {}

    for _p, _label, _flag in ((a.train, "训练集", "--train"), (a.data, "输入数据", "--data")):
        if not os.path.exists(_p):
            sys.exit(
                "找不到%s：%s\n"
                "  本脚本需要「含真值的训练集」。请确认你解压的是**完整包**，或用 %s 显式指定路径，例如：\n"
                "    python reference.py %s \"<路径>\" --out submission.jsonl"
                % (_label, _p, _flag, _flag))
    tr = load(a.train)
    rows = load(a.data)
    Xtr = [feats(r) for r in tr]
    Xte = [feats(r) for r in rows]
    rng = random.Random(a.seed)
    kw = dict(maxd=a.max_depth, minleaf=a.min_leaf, mtry=a.mtry)

    A = [r["truth"]["task_A"] for r in tr]
    B = [r["truth"]["task_B"] for r in tr]
    C = [r["truth"]["task_C"] for r in tr]

    print("[1/3] 单值任务建模（边界结论 / 严重程度 / 平台责任）", flush=True)
    tr_bnd = forest(Xtr, [x["boundary_conclusion"] for x in A], a.ntree_single, rng, **kw)
    tr_sev = forest(Xtr, [x["severity"] for x in A], a.ntree_single, rng, **kw)
    tr_plt = forest(Xtr, [x["platform_liability"] for x in B], a.ntree_single, rng, **kw)
    tr_pub = forest(Xtr, [x["publisher_liability"] for x in B], a.ntree_single, rng, **kw)
    tr_org = forest(Xtr, [x["organizer_liability"] for x in B], a.ntree_single, rng, **kw)
    tr_og = forest(Xtr, [str(bool(x["organized_infringement"])) for x in B],
                   a.ntree_single, rng, **kw)
    tr_ful = forest(Xtr, [str(bool(x["platform_obligation_fulfilled"])) for x in B],
                    a.ntree_single, rng, **kw)

    print("[2/3] 多标签任务建模（侵权类型 / 平台义务 / 处置措施）", flush=True)

    def multi_trees(vals):
        labs = sorted({v for row in vals for v in row})
        out = {}
        for lb in labs:
            y = [1 if lb in row else 0 for row in vals]
            if 0 < sum(y) < len(y):
                out[lb] = forest(Xtr, y, a.ntree_multi, rng, **kw)
        return out

    tA = multi_trees([x["infringement_types"] for x in A])
    tB = multi_trees([x["platform_obligation"] for x in B])
    tC = multi_trees([x["disposal_measures"] for x in C])

    print("[3/3] 生成提交（%.0fs）" % (time.time() - t0), flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, x in zip(rows, Xte):
            bnd = vote_single(tr_bnd, x)
            sev = vote_single(tr_sev, x)
            types = sorted(lb for lb, ts in tA.items() if vote_multi(ts, x, THR))
            if bnd in ("不构成侵权", "存疑待查"):
                types = []
            empty = not types
            if empty:
                sev = "不适用"
            in_scope = {e for t in types for e in ELEM_OF.get(t, [])}
            ej = {e: (("构成" if e in NEC else "存疑") if e in in_scope else "不适用") for e in ELEMS}
            obl = sorted(lb for lb, ts in tB.items() if vote_multi(ts, x, THR)) or \
                sorted(lb for lb, ts in tB.items())[:1]
            meas = sorted(lb for lb, ts in tC.items() if vote_multi(ts, x, THR))
            law = sorted({c for t in types for c in LAW_OF.get(t, [])})
            # 引用只取「认定类型」对应的法条/锚点。实测把候选（低置信）类型也纳入引用
            # 会因精确率下降更多而拉低引用有效率（micro-F1），故不做扩召回。
            cites = sorted(set(law) | {c for t in types for c in ANC_OF.get(t, [])})
            # 门限项「数据溯源率」= 结论覆盖率 × 引用有效率，其中**覆盖率是召回侧指标**：
            # 该给引用的样本却给了空列表，会直接丢掉覆盖率；多给只影响精确率。
            # 因此边界误判为「不构成侵权」时，仍按**最高票类型**给出引用，避免白丢覆盖率。
            if not cites:
                top1 = max(tA, key=lambda lb: sum(_walk(t, x).p for t in tA[lb]) / len(tA[lb]))
                cites = sorted(set(LAW_OF.get(top1, [])) | set(ANC_OF.get(top1, [])))
            pub = vote_single(tr_pub, x)
            org = vote_single(tr_org, x)
            plt = vote_single(tr_plt, x)
            # ★ organized_infringement 与 organizer_liability **严格等价**
            #   （真值 valid 1400 条实测 0 违例）。原实现二者各自独立建模，
            #   产出「organized_infringement=false 但 organizer_liability 非『不承担责任』」
            #   的自相矛盾（check_submission 报 35 条 ! 提示）。
            #   修法：由 org 派生，保证语义自洽。
            og = (org != "不承担责任")
            ful = vote_single(tr_ful, x) == "True"
            if empty:
                pub, org, plt, og = "不承担责任", "不承担责任", "不承担责任", False
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "prediction": {
                    "task_A": {"infringement_types": types, "boundary_conclusion": bnd,
                               "exemptions": [], "element_judgments": ej, "severity": sev},
                    "task_B": {"publisher_liability": pub, "organizer_liability": org,
                               "platform_obligation": obl,
                               "platform_obligation_fulfilled": ful,
                               "platform_liability": plt,
                               "organized_infringement": og},
                    "task_C": {"disposal_measures": [] if empty else meas,
                               "priority_order": [] if empty else _urgency(
                                   meas, r["context"]["propagation_level"], URG, UMOD)},
                    "citations": cites,
                }}, ensure_ascii=False) + "\n")
    print("已写出 %s（%d 条）。提交前请跑 check_submission.py 自检。总耗时 %.0fs"
          % (a.out, len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
