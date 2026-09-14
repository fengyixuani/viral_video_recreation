"""通用爆款复刻 Agent 的编排层：定义步骤顺序、跑步骤、断点续跑。

一个任务 = output/tasks/{task_id}/：参数、中间产物、状态、日志全部落在任务目录里。
每个步骤单独记状态，重跑时已完成的步骤直接复用产物（幂等 + 断点续跑）。

步骤与落地模块：
- ingest    素材入库校验（本模块）
- reference analyze_reference.analyze() 拆解爆款参考片
- audio     ref_audio.step_audio 判参考片音轨、按需分离 BGM 与纯人声
- materials analyze_materials.analyze() 把用户素材切成可召回片段（本模块）
- product   product_facts.step_product（商品图链路在 product_images.py）
- script    write_script.build(mode=strict|creative)（本模块）
- match     shot_match.step_match 分镜对素材，按匹配度分三路
- generate  segment_build.step_generate 逐段出片，段内按镜头混合
- compose   compose_video.step_compose 拼接 + 音轨 + 字幕

判定口径全部写在 rules.py 的规则表里（跑 python3 rules.py 看总表），这里只做执行。

复刻参数 options：
- copy_bgm 默认 True：把参考片音乐铺到成片下面（有独立 BGM 文件时优先用它）
- copy_subtitles：None（默认）跟随参考片有无字幕 / False 完全不烧 / "force" 强制烧
- 成片人声一律是 voice_dub 克隆音色后 TTS 出来的，绝不整轨照搬参考片原声
- rewrite_mode 默认 creative（允许大模型改写剧本）；strict 只替换人物/商品/场景。
  命令行入口不吃这个默认值：它总是显式传 strict，要创意模式得加 --creative
- dub_all_cuts 默认 False：直接用的用户素材按「用户切片处理」表逐条判音轨（口播对得上
  又没 BGM 的保留原声）。开成 True 时一律静音 + 克隆音色重配，整片只剩一把嗓子
- 传了商品视频就一定切片并优先复用，没有开关：素材是用户自己的实拍，比生成的可信
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import time

import analyze_materials  # pyright: ignore[reportImplicitRelativeImport]
import analyze_reference  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from Agent_tools import registry as agent_tools
from compose_video import step_compose
from product_facts import step_product
from ref_audio import step_audio
from segment_build import step_generate
from shot_match import step_match
# 下面这些是 re-export：server.py / review_case.py 以及历史脚本都按 pipeline.xxx 调用，
# 实现搬到 task_store.py / media.py / segment_build.py / compose_video.py 后调用方不用改。
from compose_video import _cjk_font, _pieces  # noqa: F401
from media import _ffmpeg, _probe  # noqa: F401
from segment_build import _cuttable  # noqa: F401
from task_store import (DEFAULT_OPTIONS, IMAGE_EXTS, KIND_EXTS, LIBRARY_DIR,  # noqa: F401
                        TASKS_DIR, VIDEO_EXTS, _LOCK, _d, _p, _rel, adopt_reusable, cancel_task,
                        create_task, delete_task, file_md5, find_reusable_file,
                        find_reusable_video, is_cancelled, is_reusable, list_library, list_tasks,
                        load, log, pick_input, register_input, remove_input, save, task_dir,
                        TaskCancelled, uncancel_task, update_task)

STEP_TITLES = {
    "ingest": "素材入库校验",
    "reference": "爆款参考片拆解",
    "audio": "参考片音频判定",
    "product": "商品事实卡",
    "materials": "用户素材切片",
    "script": "剧本与分段",
    "match": "分镜素材匹配",
    "generate": "分段出片",
    "compose": "拼接与音轨字幕",
}

# 进度条上每步只放得下三四个字，完整名字留给 tooltip
STEP_SHORT = {
    "ingest": "入库", "reference": "拆解", "audio": "音频", "product": "事实卡",
    "materials": "切片", "script": "剧本", "match": "匹配", "generate": "出片",
    "compose": "合成",
}

# ---------------- 步骤 1：素材入库 ----------------
def step_ingest(rec: dict) -> dict:
    ins = rec["inputs"]
    if not ins.get("reference_video") or not os.path.isfile(ins["reference_video"]):
        raise ValueError("缺少爆款参考视频")
    if not (rec["product"].get("name") or ins.get("product_images") or ins.get("user_videos")):
        raise ValueError("请至少提供商品名称、商品图或用户素材视频")
    registry = {"reference_video": _probe(ins["reference_video"]),
                "product_images": [_probe(p) for p in ins.get("product_images") or []],
                "user_videos": [_probe(p) for p in ins.get("user_videos") or []],
                "person_images": [{"file": p, "name": os.path.basename(p),
                                   "size_mb": round(os.path.getsize(p) / (1 << 20), 2)}
                                  for p in ins.get("person_images") or []],
                "bgm": _probe(ins["bgm"]) if ins.get("bgm") else None}
    path = _p(rec["task_id"], "assets", "registry.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)
    tools = agent_tools.probe()
    log(rec, "外挂工具：声音克隆%s，字幕风格克隆%s，词级ASR%s，补片人声校对%s"
        % tuple("可用" if tools[k] else "未就绪"
                for k in ("tts_clone", "caption_clone", "caption_asr", "firered_asr")))
    # 补片人声校对的模型加载要十几秒，而它到 generate 步才用得上。这里就把常驻 worker
    # 拉起来（非阻塞），等用到时模型已经热了，不占 generate 的时间。
    if tools["firered_asr"]:
        agent_tools.prewarm_asr()
    log(rec, "参考片 %.1fs，商品图 %d 张，用户素材 %d 条，人物参考图 %d 张"
        % (registry["reference_video"]["duration_sec"], len(registry["product_images"]),
           len(registry["user_videos"]), len(registry["person_images"])))
    return {"artifact": _rel(rec["task_id"], path),
            "reference_duration_sec": registry["reference_video"]["duration_sec"]}


# ---------------- 步骤 2：参考片拆解 ----------------
def step_reference(rec: dict) -> dict:
    out = _p(rec["task_id"], "reference", "analysis.json")
    analysis = analyze_reference.analyze(rec["inputs"]["reference_video"])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=2)
    with open(_p(rec["task_id"], "reference", "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(analyze_reference.to_markdown(analysis))
    log(rec, "拆出 %d 个分镜（链路 %s）" % (len(analysis["分镜"]), analysis.get("链路")))
    return {"artifact": _rel(rec["task_id"], out), "shots": len(analysis["分镜"]),
            "duration_sec": analysis.get("总时长秒")}


# ---------------- 步骤 4：用户素材切片 ----------------
def step_materials(rec: dict) -> dict:
    videos = rec["inputs"].get("user_videos") or []
    # 传了素材就一定要理解并切片：素材是用户自己的商品实拍，比模型生成的可信，
    # 而且没有商品图时商品事实卡还要靠这些片段抽商品帧，没有开关可关。
    index = {"启用": bool(videos), "素材": [], "片段": []}
    if index["启用"]:
        def one(path):
            out = analyze_materials.analyze(path)
            for seg in out["片段"]:
                seg["源文件"] = path
            return out

        with cf.ThreadPoolExecutor(min(3, len(videos))) as ex:
            for out in ex.map(one, videos):
                # 「主角人声」必须一起带出来：它是 voice_dub 挑克隆基准音的第一优先候选，
                # 留在 analyze() 的返回值里就丢了（下游只读 material_index.json）
                index["素材"].append({k: out.get(k) for k in
                                      ("素材ID", "视频", "总时长秒", "整体", "主角人声")})
                index["片段"].extend(out["片段"])
    path = _p(rec["task_id"], "assets", "material_index.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    log(rec, "用户素材片段 %d 个（%s）"
        % (len(index["片段"]), "已切片" if index["启用"] else "没有商品视频"))
    return {"artifact": _rel(rec["task_id"], path), "segments": len(index["片段"])}


# ---------------- 步骤 5：剧本与分段 ----------------
def step_script(rec: dict) -> dict:
    tid = rec["task_id"]
    with open(_p(tid, "product", "fact_card.json"), encoding="utf-8") as fh:
        product = json.load(fh)
    built = write_script.build(_p(tid, "reference", "analysis.json"), product,
                              with_images=True, mode=rec["options"]["rewrite_mode"],
                              outdir=_p(tid, "script"),
                              person_images=rec["inputs"].get("person_images") or None,
                              narrative_gate=rec["options"].get("narrative_gate", True))
    built.pop("outdir", None)
    built["复刻强度"] = rec["options"]["rewrite_mode"]
    path = _p(tid, "script", "script.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(built, fh, ensure_ascii=False, indent=2)
    with open(_p(tid, "script", "script.md"), "w", encoding="utf-8") as fh:
        fh.write(write_script.to_markdown(built))
    log(rec, "剧本 %d 镜 / %d 段（%s）" % (len(built["剧本"].get("分镜") or []),
                                          len(built["分段"]), rec["options"]["rewrite_mode"]))
    audit = built["剧本"].get("商品状态审查") or {}
    if audit:
        log(rec, "商品状态审查：%s" % audit.get("状态"))
    tale = built["剧本"].get("叙事连贯审查") or {}
    if tale:
        fix = tale.get("修补") or {}
        log(rec, "叙事连贯审查：%s（可理解性 %s%s），盲测看成「%s」"
            % (tale.get("状态"), tale.get("可理解性", "-"),
               "，自动修补 %s，首轮 %s 分" % (fix.get("状态"),
                                             (tale.get("首轮") or {}).get("可理解性", "-"))
               if fix.get("已改") else "",
               tale.get("盲测概要") or "-"))
    for warn in built["剧本"].get("剧本告警") or []:
        log(rec, "剧本告警：%s" % warn)
    if built.get("拦下"):
        # 剧本与两层审查痕迹已经落盘（script.json / script.md / report.md），任务停在这一步，
        # 一张设定图都没出、一段都没生成。人看过审查结论后可以改商品信息/换参考片再
        # 「从 script 重跑」，或把 options.narrative_gate 关掉强行往下走。
        log(rec, "已拦下：%s" % built["拦下"])
        raise ValueError(built["拦下"])
    return {"artifact": _rel(tid, path), "shots": len(built["剧本"].get("分镜") or []),
            "segments": len(built["分段"])}


# ---------------- 执行器 ----------------
# materials 必须排在 product 前面：没有商品图时，商品事实卡要靠素材切片的结论去抽商品帧
# 剧本先行原则：write_script 只看参考片拆解与商品事实卡，不看素材库——先定剧本，
# 再让素材向剧本靠（match 裁剪/素材编辑、shot_refs 状态图改造）；素材做不到的记「缺失」，
# 靠提示词还原，不反过来迁就素材改剧本。只有用户视频素材很丰富、多数分镜有真画面可用时，
# 才值得考虑轻度调整剧本换取素材利用率（目前没有这条通路，加它之前先确认这个前提）。
STEPS = [("ingest", step_ingest), ("reference", step_reference), ("audio", step_audio),
         ("materials", step_materials), ("product", step_product), ("script", step_script),
         ("match", step_match), ("generate", step_generate), ("compose", step_compose)]

# 阶段并行：reference（拆参考片）与 materials（切用户素材）互不依赖；audio 要读
# reference 的分析结果，product 要吃 materials 的切片结论（无商品图时抽帧），
# 所以 audio ∥ product 排在下一阶段。并行阶段共享同一个 rec，写状态/日志/落盘由
# task_store._LOCK 串行化。
STAGES = [["ingest"], ["reference", "materials"], ["audio", "product"], ["script"],
          ["match"], ["generate"], ["compose"]]


def _run_step(rec: dict, name: str, fn) -> dict:
    done = rec["steps"].get(name) or {}
    if done.get("status") == "done":
        log(rec, "跳过 %s（已完成，复用产物）" % STEP_TITLES[name])
        return done
    # 步骤边界先看一眼是不是已经被要求终止，别再白跑一步、白花模型额度
    if is_cancelled(rec["task_id"]):
        raise TaskCancelled("任务已终止：%s" % rec["task_id"])
    with _LOCK:
        rec.update({"status": "running", "step": name})
        # started_at 给人看，started_ts 给 server._fill_elapsed 算「这一步已经跑了多久」——
        # 只写 started_at 的话那个函数的条件永远为假，进度条上运行中步骤的耗时一直空白。
        rec["steps"][name] = {"status": "running", "started_at": time.strftime("%H:%M:%S"),
                              "started_ts": time.time()}
    log(rec, "开始 %s" % STEP_TITLES[name])
    save(rec)
    start = time.time()
    try:
        out = fn(rec) or {}
    except Exception as exc:  # noqa: BLE001
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


def _build_report(task_id: str) -> None:
    """出一份 report.md（每张图/每段视频的原料、prompt、产物）。失败任务也出：
    报告只读已落盘的中间产物，跑了一半的任务正好靠它定位。报告失败绝不拖垮流水线。"""
    try:
        import report  # noqa: PLC0415  延迟导入，避免与本模块互相引用
        path = report.build(task_id)
        rec = load(task_id)
        log(rec, "生成报告：%s" % _rel(task_id, path))
        save(rec)
    except Exception as exc:  # noqa: BLE001
        print("[%s] 报告生成失败：%s" % (task_id, str(exc)[:200]), flush=True)


INPUT_LABELS = (("reference_video", "参考片"), ("product_images", "商品图"),
                ("user_videos", "自有素材"), ("person_images", "人物图"), ("bgm", "背景音乐"))


def _log_inputs(rec: dict) -> None:
    """开跑时把这次真正用到的素材一次性写进日志。

    登记那会儿（上传/从素材库挑）不写：挑素材经常反复换，一条条记下来会把日志刷满，
    还看不出最后进这次复刻的是哪几个文件。名字多了只列前几个，全量看素材区。
    """
    for kind, label in INPUT_LABELS:
        cur = rec["inputs"].get(kind)
        files = [os.path.basename(p) for p in
                 (list(cur) if isinstance(cur, list) else ([cur] if cur else []))]
        if not files:
            continue
        head = "、".join(files[:4]) + ("…等 %d 个" % len(files) if len(files) > 4 else "")
        log(rec, "%s%s：%s" % (label, "（%d）" % len(files) if len(files) > 1 else "", head))


def run(task_id: str) -> dict:
    """跑完整条流水线；已完成的步骤自动跳过，失败时任务停在该步骤。"""
    rec = load(task_id)
    rec["error"] = ""
    _log_inputs(rec)
    save(rec)
    fns = dict(STEPS)
    try:
        for stage in STAGES:
            todo = [n for n in stage
                    if (rec["steps"].get(n) or {}).get("status") != "done"]
            for name in stage:                      # 已完成的照旧打「跳过」日志
                if name not in todo:
                    _run_step(rec, name, fns[name])
            if len(todo) == 1:
                _run_step(rec, todo[0], fns[todo[0]])
            elif todo:
                with cf.ThreadPoolExecutor(len(todo)) as ex:
                    for f in [ex.submit(_run_step, rec, n, fns[n]) for n in todo]:
                        f.result()
    except TaskCancelled:
        # 目录可能已经被删了，别再去读盘出报告
        print("[%s] 任务已终止，流水线退出" % task_id, flush=True)
        return {"task_id": task_id, "status": "cancelled"}
    except Exception:  # noqa: BLE001  失败详情已写进 task.json
        _build_report(task_id)
        return load(task_id)
    rec.update({"status": "completed", "step": ""})
    save(rec)
    _build_report(task_id)
    return load(task_id)


# 每步的主产物：reset_from 重跑前把它们挪进 history/ 留档，新一轮从零写，
# 旧结果永远可回看（用户要求：重新生成不覆盖之前的结果）。可以是目录也可以是单个文件。
# 只列这一步自己写出来的产物：assets/uploads/{kind}/ 是任务的输入素材（参考片、商品图、用户
# 视频），任何步骤重跑都不能动它——挪走后 inputs 里的路径全指向 history/，任务再也读不到
# 自己的素材。registry.json 归 ingest、material_index.json 归 materials，各自重跑各自留档。
STEP_DIRS = {"ingest": ["assets/registry.json"], "reference": ["reference"], "audio": ["audio"],
             "materials": ["assets/material_index.json"], "product": ["product"],
             "script": ["script"], "match": ["edit"], "generate": ["generated"],
             "compose": ["render"]}


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
            parts = sub.split("/")
            src = os.path.join(task_dir(task_id), *parts)
            if not os.path.exists(src) or (os.path.isdir(src) and not os.listdir(src)):
                continue
            dst = os.path.join(task_dir(task_id), "history", stamp, *parts)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            kept.append(sub)
    report = os.path.join(task_dir(task_id), "report.md")
    if kept and os.path.isfile(report):     # 旧报告跟着旧产物一起留档
        os.rename(report, os.path.join(task_dir(task_id), "history", stamp, "report.md"))
    rec.update({"status": "draft", "step": "", "error": ""})
    rec.pop("result", None)
    log(rec, "已重置步骤 %s 及其后续%s" % (STEP_TITLES[step],
        "，旧产物留档 history/%s/（%s）" % (stamp, "、".join(kept)) if kept else ""))
    return save(rec)


def _case_paths(tag: str) -> tuple:
    """按仓库约定找一个用例的素材与参考片：materials/<编号>_* 与 videos/others/<编号>_*.mp4。"""
    root = os.path.dirname(os.path.abspath(__file__))
    mats = [d for d in sorted(glob.glob(os.path.join(root, "materials", tag + "*")))
            if os.path.isdir(d)]
    refs = sorted(glob.glob(os.path.join(root, "videos", "others", tag + "*.mp4")))
    if not mats:
        raise SystemExit("找不到素材目录：materials/%s*" % tag)
    if not refs:
        raise SystemExit("找不到参考片：videos/others/%s*.mp4" % tag)
    return mats[0], refs[0]


def main(argv=None):
    """CLI：python3 pipeline.py <用例编号> [--name 商品名] [--point 卖点]…

    正式入口是 server.py 的 Web 界面，这个 CLI 只为跑用例方便。
    --task 配合 --from 可以在已有任务上从某一步重跑（产物留着当备份）。
    """
    ap = argparse.ArgumentParser(description="跑一条爆款复刻任务")
    ap.add_argument("case", help="用例编号，如 17 / 19 / 24（按 materials/<编号>_* 找素材）")
    ap.add_argument("--name", default="", help="商品名，默认取素材目录名后缀")
    ap.add_argument("--category", default="", help="品类")
    ap.add_argument("--point", action="append", default=[], help="卖点，可重复")
    ap.add_argument("--task", default="", help="在已有任务上继续/重跑")
    ap.add_argument("--from", dest="from_step", default="",
                    help="配合 --task：从这一步开始重跑（%s）" % "/".join(n for n, _ in STEPS))
    ap.add_argument("--no-bgm", action="store_true", help="不复刻参考片音乐")
    ap.add_argument("--subtitles", action="store_true", help="强制烧字幕")
    ap.add_argument("--no-subtitles", action="store_true", help="强制不烧字幕（默认跟随参考片）")
    ap.add_argument("--dub-all-cuts", action="store_true",
                    help="用户素材一律静音后用克隆音色重配（默认按规则表逐条判，可保原声）")
    ap.add_argument("--creative", action="store_true", help="creative 模式（允许改写剧本）")
    args = ap.parse_args(argv)

    if args.task:
        if args.from_step:
            reset_from(args.task, args.from_step)
        task_id = args.task
    else:
        mats, ref = _case_paths(args.case)
        images = sorted(p for e in IMAGE_EXTS
                        for p in glob.glob(os.path.join(mats, "*" + e)))
        videos = sorted(p for e in VIDEO_EXTS
                        for p in glob.glob(os.path.join(mats, "*" + e)))
        name = args.name or os.path.basename(mats).split("_", 1)[-1]
        product = {"name": name, "category": args.category,
                   "selling_points": args.point}
        options = {"copy_bgm": not args.no_bgm,
                   "copy_subtitles": ("force" if args.subtitles
                                      else False if args.no_subtitles else None),
                   "dub_all_cuts": args.dub_all_cuts,
                   "rewrite_mode": "creative" if args.creative else "strict"}
        # 任务目录用可读名：日期_用例_参考X生成Y（重名由 create_task 自动加 _2/_3）
        ref_topic = os.path.splitext(os.path.basename(ref))[0].split("_", 1)[-1]
        mat_topic = os.path.basename(mats).split("_", 1)[-1]
        slug = ("%s_%s_参考%s生成%s" % (time.strftime("%Y%m%d"), args.case,
                                        ref_topic, mat_topic)
                + ("_creative" if args.creative else ""))
        rec = create_task(product, options, title=name + " 复刻", slug=slug)
        task_id = rec["task_id"]
        register_input(task_id, "reference_video", ref)
        for p in images:
            register_input(task_id, "product_images", p)
        for p in videos:
            register_input(task_id, "user_videos", p)
        print("用例 %s：参考片 %s，素材 %s（图 %d 张，视频 %d 条）"
              % (args.case, os.path.basename(ref), os.path.basename(mats),
                 len(images), len(videos)), flush=True)

    out = run(task_id)
    print("状态：%s" % out["status"])
    print("任务目录：%s" % task_dir(out["task_id"]))
    if out.get("result"):
        print("成片：%s" % os.path.join(task_dir(out["task_id"]), out["result"]["final"]))
    return 0 if out["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
