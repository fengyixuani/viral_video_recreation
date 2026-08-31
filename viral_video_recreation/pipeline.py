# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""编排层：定义步骤顺序、跑步骤、断点续跑。

一个任务 = <OUTPUT_DIR>/tasks/{task_id}/：参数、中间产物、状态、日志全落在任务目录里。
每个步骤单独记状态，重跑时已完成的步骤直接复用产物（幂等 + 断点续跑）。

当前只落地了基础步骤 ingest（素材入库校验）。后续能力按下面的顺序往 STEPS/STAGES 里
接：reference（参考片拆解）→ audio（音频判定）/ materials（素材切片）→ product（商品
事实卡）→ script（剧本分段）→ match（分镜素材匹配）→ generate（分段出片）→ compose
（拼接与音轨字幕）。执行器与断点续跑不需要跟着改，只加步骤函数和 STEP_DIRS 条目。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import concurrent.futures as cf
import json
import os
import time

from . import media
from .task_store import (_LOCK, load, log, rel, save, task_dir, task_path,
                         task_subdir)

STEP_TITLES = {
    "ingest": "素材入库校验",
}


# ---------------- 步骤 1：素材入库 ----------------
def step_ingest(rec: dict) -> dict:
    """校验输入是否够开工，并把素材基础事实落成 assets/registry.json。"""
    ins = rec["inputs"]
    if not ins.get("reference_video") or not os.path.isfile(ins["reference_video"]):
        raise ValueError("缺少爆款参考视频")
    if not (rec["product"].get("name") or ins.get("product_images")
            or ins.get("user_videos")):
        raise ValueError("请至少提供商品名称、商品图或用户素材视频")
    registry = {
        "reference_video": media.probe(ins["reference_video"]),
        "product_images": [media.probe(p) for p in ins.get("product_images") or []],
        "user_videos": [media.probe(p) for p in ins.get("user_videos") or []],
        "person_images": [{"file": p, "name": os.path.basename(p),
                           "size_mb": round(os.path.getsize(p) / (1 << 20), 2)}
                          for p in ins.get("person_images") or []],
        "bgm": media.probe(ins["bgm"]) if ins.get("bgm") else None,
    }
    path = task_path(rec["task_id"], "assets", "registry.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)
    log(rec, "参考片 %.1fs，商品图 %d 张，用户素材 %d 条，人物参考图 %d 张"
        % (registry["reference_video"]["duration_sec"], len(registry["product_images"]),
           len(registry["user_videos"]), len(registry["person_images"])))
    return {"artifact": rel(rec["task_id"], path),
            "reference_duration_sec": registry["reference_video"]["duration_sec"]}


# ---------------- 执行器 ----------------
STEPS = [("ingest", step_ingest)]

# 阶段内的步骤并行跑，阶段之间串行。并行阶段共享同一个 rec，写状态/日志/落盘
# 由 task_store._LOCK 串行化。
STAGES = [["ingest"]]

# 每步的主产物目录：reset_from 重跑前把这些目录挪进 history/ 留档，
# 新一轮从零写，旧结果永远可回看（重新生成不覆盖之前的结果）。
STEP_DIRS = {"ingest": []}


def _run_step(rec: dict, name: str, fn) -> dict:
    """跑一个步骤：已完成的直接复用，失败把原因写进 task.json 后原样抛出。"""
    done = rec["steps"].get(name) or {}
    if done.get("status") == "done":
        log(rec, "跳过 %s（已完成，复用产物）" % STEP_TITLES[name])
        return done
    with _LOCK:
        rec.update({"status": "running", "step": name})
        rec["steps"][name] = {"status": "running", "started_at": time.strftime("%H:%M:%S")}
    log(rec, "开始 %s" % STEP_TITLES[name])
    save(rec)
    start = time.time()
    try:
        out = fn(rec) or {}
    except Exception as exc:   # noqa: BLE001  失败详情写进 task.json 后原样抛出
        with _LOCK:
            rec["steps"][name] = {"status": "failed", "sec": round(time.time() - start, 1),
                                  "error": str(exc)[:600]}
            rec.update({"status": "failed",
                        "error": "%s：%s" % (STEP_TITLES[name], str(exc)[:300])})
        log(rec, "步骤失败 %s：%s" % (STEP_TITLES[name], str(exc)[:200]))
        save(rec)
        raise
    with _LOCK:
        rec["steps"][name] = dict(out, status="done", sec=round(time.time() - start, 1))
    log(rec, "完成 %s（%.1fs）" % (STEP_TITLES[name], rec["steps"][name]["sec"]))
    return save(rec)["steps"][name]


def run(task_id: str) -> dict:
    """跑完整条流水线；已完成的步骤自动跳过，失败时任务停在该步骤。"""
    rec = load(task_id)
    rec["error"] = ""
    fns = dict(STEPS)
    try:
        for stage in STAGES:
            todo = [n for n in stage
                    if (rec["steps"].get(n) or {}).get("status") != "done"]
            for name in stage:                 # 已完成的照旧打「跳过」日志
                if name not in todo:
                    _run_step(rec, name, fns[name])
            if len(todo) == 1:
                _run_step(rec, todo[0], fns[todo[0]])
            elif todo:
                with cf.ThreadPoolExecutor(len(todo)) as pool:
                    for fut in [pool.submit(_run_step, rec, n, fns[n]) for n in todo]:
                        fut.result()
    except Exception:   # noqa: BLE001  失败详情已写进 task.json
        return load(task_id)
    rec.update({"status": "completed", "step": ""})
    save(rec)
    return load(task_id)


def reset_from(task_id: str, step: str) -> dict:
    """从某个步骤开始重跑：清掉它及其后续步骤的状态，旧产物挪进 history/ 留档。"""
    rec = load(task_id)
    names = [n for n, _ in STEPS]
    if step not in names:
        raise ValueError("未知步骤：%s" % step)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kept = []
    for name in names[names.index(step):]:
        rec["steps"].pop(name, None)
        for sub in STEP_DIRS.get(name, ()):
            src = task_subdir(task_id, sub)
            if os.path.isdir(src) and os.listdir(src):
                dst = os.path.join(task_dir(task_id), "history", stamp, sub)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                kept.append(sub)
    rec.update({"status": "draft", "step": "", "error": ""})
    rec.pop("result", None)
    log(rec, "已重置步骤 %s 及其后续%s"
        % (STEP_TITLES[step],
           "，旧产物留档 history/%s/（%s）" % (stamp, "、".join(kept)) if kept else ""))
    return save(rec)
