# -*- coding: utf-8 -*-
"""提交格式自检工具（不读真值，可随时运行）。

用法：
    python check_submission.py --pred my_submission.jsonl --data "../02_选手数据集/初赛_无真值/prelim.jsonl"
    python check_submission.py --pred my_submission.jsonl --data "../02_选手数据集/初赛_无真值/prelim.jsonl" --labels ../schema/labels.json

退出码：存在 ERROR 时为 1，否则为 0（仅有 WARN 不影响退出码）。
"""
import json
import os
import sys
import argparse


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append((i, json.loads(line)))
            except Exception as e:
                sys.exit("第 %d 行不是合法 JSON：%s" % (i, e))
    return rows


def _urgency(codes, prop_level, base_order, modifier):
    """按「基础紧迫度序 + 案情档位修正」生成期望的 priority_order。

    与 label 体系（labels.json.disposal_urgency_modifier）完全一致：
    当样本 propagation_level 命中触发档时，把 promote 列出的措施提权至最前。
    """
    seq = sorted(codes, key=lambda m: base_order.index(m) if m in base_order else 999)
    if modifier and prop_level is not None and modifier.get("trigger") == prop_level:
        pro = [m for m in seq if m in (modifier.get("promote") or [])]
        rest = [m for m in seq if m not in (modifier.get("promote") or [])]
        return pro + rest
    return seq


def _vals(x):
    """把 labels.json 里的取值容器（list 或 dict）统一成取值集合。"""
    if isinstance(x, dict):
        out = set()
        for k, v in x.items():
            out.add(k)
            if isinstance(v, str):
                out.add(v)
        return out
    if isinstance(x, list):
        out = set()
        for v in x:
            if isinstance(v, dict):
                out.add(v.get("code") or v.get("name") or json.dumps(v, ensure_ascii=False))
            else:
                out.add(v)
        return out
    return {x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="待自检的提交文件（jsonl）")
    ap.add_argument("--data", default="", help="对应输入文件（jsonl），用于校验 sample_id 覆盖")
    ap.add_argument("--labels", default="", help="labels.json 路径（默认 ../schema/labels.json）")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    lab_path = a.labels or os.path.join(here, "..", "schema", "labels.json")
    lab = json.load(open(lab_path, encoding="utf-8"))

    TYPES = {t["code"] for t in lab["infringement_types"]}
    ELEMS = set(lab["elements"])
    EVALS = set(lab["element_values"])
    BND = set(lab["boundary_values"])
    SEV = set(lab["severity"])
    EXM = set(lab["exemptions"])
    MEAS = {m["code"] for m in lab["disposal_measures"]}
    URG = list(lab["disposal_urgency_order"])
    UMOD = lab.get("disposal_urgency_modifier") or {}
    LAW = set(lab["law_articles"])
    ANC = set(lab["case_anchors"])
    LIAB = lab.get("liability") or {}
    PUB = _vals(LIAB.get("publisher"))
    ORG = _vals(LIAB.get("organizer"))
    PLT = _vals(LIAB.get("platform_type"))
    OBL = _vals(LIAB.get("platform_obligation"))

    # 输入侧 propagation_level（用于校验优先级的「案情档位修正」）
    PLV = {}
    if a.data:
        for l in open(a.data, encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                PLV[o.get("sample_id")] = ((o.get("context") or {}) or {}).get("propagation_level")

    rows = load_jsonl(a.pred)
    errors, warns = [], []
    ids = []
    for ln, r in rows:
        if not isinstance(r, dict):
            errors.append("第 %d 行不是对象" % ln); continue
        sid = r.get("sample_id")
        if not sid:
            errors.append("第 %d 行缺少 sample_id" % ln); continue
        ids.append(sid)
        pred = r.get("prediction")
        if not isinstance(pred, dict):
            errors.append("%s 缺少 prediction 对象" % sid); continue
        A = pred.get("task_A") or {}
        B = pred.get("task_B") or {}
        C = pred.get("task_C") or {}
        # ★ 三个子任务均须提交：整块缺失给一条明确 ERROR（否则会退化成
        #   「publisher_liability 非法：None」这类看不出所以然的报错）。
        for _k, _lab in (("task_A", "A"), ("task_B", "B"), ("task_C", "C")):
            if not isinstance(pred.get(_k), dict) or not pred.get(_k):
                errors.append("%s 未提交子任务 %s（三个子任务均须提交）" % (sid, _lab))

        ts = A.get("infringement_types") or []
        bad = [t for t in ts if t not in TYPES]
        if bad:
            errors.append("%s 侵权类型编码非法：%s" % (sid, bad))
        bnd = A.get("boundary_conclusion")
        if bnd not in BND:
            errors.append("%s boundary_conclusion 非法：%r" % (sid, bnd))
        if bnd in ("不构成侵权", "存疑待查") and ts:
            errors.append("%s 边界结论为「%s」但仍给出侵权类型" % (sid, bnd))
        for e in (A.get("exemptions") or []):
            if e not in EXM:
                errors.append("%s 免责事由编码非法：%s" % (sid, e))
        ej = A.get("element_judgments") or {}
        if not ej:
            errors.append("%s 未输出 element_judgments（要件级 F1 将直接为 0）" % sid)
        else:
            for k, v in ej.items():
                if k not in ELEMS:
                    warns.append("%s 未知要素编码：%s" % (sid, k))
                if v not in EVALS:
                    errors.append("%s 要素 %s 取值非法：%r" % (sid, k, v))
        if A.get("severity") not in SEV:
            errors.append("%s severity 非法：%r" % (sid, A.get("severity")))

        if B and B.get("publisher_liability") not in PUB and PUB:
            errors.append("%s publisher_liability 非法：%r" % (sid, B.get("publisher_liability")))
        if B and B.get("organizer_liability") not in ORG and ORG:
            errors.append("%s organizer_liability 非法：%r" % (sid, B.get("organizer_liability")))
        if B and B.get("platform_liability") not in PLT and PLT:
            errors.append("%s platform_liability 非法：%r" % (sid, B.get("platform_liability")))
        ob = B.get("platform_obligation") or []
        if not isinstance(ob, list) or not ob:
            warns.append("%s platform_obligation 为空或非列表" % sid)
        else:
            for o in ob:
                if OBL and o not in OBL:
                    errors.append("%s 平台义务取值非法：%s" % (sid, o))
        og = B.get("organized_infringement")
        if not isinstance(og, bool):
            warns.append("%s organized_infringement 不是布尔值：%r" % (sid, og))
        elif og and B.get("organizer_liability") in ("不承担责任", None):
            warns.append("%s organized_infringement=true 但 organizer_liability 未体现组织者责任" % sid)
        elif (not og) and B.get("organizer_liability") not in ("不承担责任", None):
            warns.append("%s organized_infringement=false 但 organizer_liability 非「不承担责任」" % sid)

        ms = C.get("disposal_measures") or []
        bad = [m for m in ms if m not in MEAS]
        if bad:
            errors.append("%s 处置措施编码非法：%s" % (sid, bad))
        if bnd in ("不构成侵权", "存疑待查") and (ms or (C.get("priority_order") or [])):
            errors.append("%s 边界样本仍给出处置措施/优先级/法条（应为空列表）" % sid)
        po = C.get("priority_order") or []
        if ms and not po:
            errors.append("%s 给了措施却未给 priority_order（Kendall τ 将计 0）" % sid)
        elif po:
            if set(po) != set(ms):
                warns.append("%s priority_order 与 disposal_measures 集合不一致" % sid)
            exp = _urgency(po, PLV.get(sid), URG, UMOD)
            if po != exp:
                warns.append("%s priority_order 未按紧迫度序列排序（期望 %s）" % (sid, exp))
        cit = pred.get("citations") or []
        if not cit:
            if ms:
                errors.append("%s 未输出 citations（须放在 `prediction` 内、与 `task_A`/`task_B`/`task_C` 同级）——门限项「数据溯源率」将计 0" % sid)
        for c in cit:
            if c not in LAW and c not in ANC:
                errors.append("%s citations 含不存在的编码：%s" % (sid, c))
        # 提示：引用有效率按「与本样本结论相关」计 micro-F1，引用条数本身不是问题，
        # 但把 labels.json 的法条表整体铺满会让精确率崩塌（详见赛题说明文档 4.4 与选手指南第五节）。

    if not a.data:
        warns.append("未提供 --data：无法校验 priority_order 的「案情档位修正」"
                     "（建议：--data 传入对应的输入 jsonl）")
    dup = len(ids) - len(set(ids))
    if dup:
        errors.append("存在 %d 个重复 sample_id" % dup)
    if a.data:
        want = {json.loads(l)["sample_id"] for l in open(a.data, encoding="utf-8") if l.strip()}
        miss = want - set(ids)
        extra = set(ids) - want
        if miss:
            errors.append("缺少 %d 条样本预测（示例：%s）" % (len(miss), sorted(miss)[:3]))
        if extra:
            errors.append("多出 %d 条不在输入文件中的 sample_id（示例：%s）" % (len(extra), sorted(extra)[:3]))

    print("提交文件        : %s" % a.pred)
    print("样本数          : %d" % len(ids))
    print("ERROR           : %d" % len(errors))
    for e in errors[:40]:
        print("  ✗ %s" % e)
    if len(errors) > 40:
        print("  … 其余 %d 条略" % (len(errors) - 40))
    print("WARN            : %d" % len(warns))
    for x in warns[:20]:
        print("  ! %s" % x)
    if len(warns) > 20:
        print("  … 其余 %d 条略" % (len(warns) - 20))
    if not errors:
        if warns:
            print("结果            : 通过（无 ERROR），但有 %d 条警告需逐条确认" % len(warns))
        else:
            print("结果            : 通过（无 ERROR、无警告）")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
