CAIL2026 涉网侵权治理初赛2100条 — v3仅排序修正版

本版按用户要求，只保留134条确定性排序修正，撤回v2中的全部10条责任判断修改。
官方checker结果：0 ERROR、10 WARN；10条均为原版责任字段一致性警告，已单独列出并原样保留，不代表新增实体判断。

一、修改边界
从不可变的原始2100条预测重建，只调整134条 prediction.task_C.priority_order。
基础紧迫度及传播档位“爆”时的M03提权均按官方labels.json执行；措施集合不增删。
全部2100条的task_B、task_A、citations及disposal_measures均与原版完全一致。
10条责任案例整行恢复原版，字节级相同。其余排序修改与v2一致。
相对原版仅134行/134字段不同，1966行字节级不变。

二、优先使用哪些文件
CAIL2026_swqqzl_Qwen38_thinking_T06_prelim2100_sortonly_v3_submission.zip：新版提交包，唯一成员prediction.jsonl，完整2100条。
prediction.jsonl / MODIFIED_FILE.jsonl：同一份新版完整结果。
OFFICIAL_WARNINGS_10.txt / .csv / .json：剩余10条WARN，包含原始checker文字；CSV含保留原因。
RESPONSIBILITY_RESTORED_10.json：10条从v2恢复到原版的具体字段、值、完整B及案件输入。
CHANGES_134.jsonl / DIFF_FILE.patch：相对原版仅134条排序改动的逐项记录和差异。
RESTORE_FROM_V2.patch：相对v2撤回10条责任修改的差异。
VERIFICATION.txt / checks：原版、v2参照、新版、独立副本回滚的命令、输入、逐字输出及退出状态。

三、原版和原始响应
original/prediction.jsonl：原版2100条，不修改。
original/REFERENCE_V2_prediction.jsonl：仅供对照的v2，不是本次提交文件。
original/ORIGINAL_SUBMISSION.zip：原始提交ZIP，仅供留档。
original/ORIGINAL_HANDOFF.zip：原始完整交接包，含2100条原生输出、thinking、请求、提示词、输入和历史失败记录，字节不变。
原版、v2及上轮责任审查目录仍在原位置保留，SHA256清单见original/ORIGINAL_FILES_SHA256.json。

四、剩余WARN
8条是 organized_infringement=false 但原organizer_liability非“不承担责任”；
2条是 organized_infringement=true 但原organizer_liability为“不承担责任”。
本版不决定这10条谁承担何种责任，也不证明原答案正确；只按用户要求恢复原判并保留警告，等待实体复核。
官方checker未改字节，完整执行；没有过滤或压制警告。

五、复验
在本目录执行 python3 VERIFY_SORTONLY.py baseline / modified / rollback。
预期依次为0 ERROR/144 WARN、0 ERROR/10 WARN、回滚副本0 ERROR/144 WARN；退出状态均为0。
ROLLBACK.sh仅允许修改rollback_test/下的独立副本，拒绝覆盖新版交付结果。

六、运行与原生性
模型仍是原运行的Qwen/Qwen3.8-27B，thinking=true、reasoning_effort=medium、temperature=0.6。
本次没有重跑模型，未改提示词、生成预算和原始响应，未上传比赛，未重启定时任务。
原2100条都与原生最终JSON相等、thinking非空、finish_reason=stop。
v3是确定性排序后处理结果：1966条仍等于原生输出，134条仅优先级数组不同。
校验通过是数据/格式/修改范围通过，不是官方准确率或比赛评分。
