#!/usr/bin/env python3
"""caption_clone —— 字幕特效克隆：给一条成片烧上「和参考视频同风格」的字幕。

从 Viral_Video_Agent 的 `src/editing/whq_clone/captions_clone` 抽出来的独立版本：
去掉了对 Agent 内部模块（`_common` / `pipeline_utils` / `as_core` / `obs`）的依赖，
LLM/VLM 走环境变量配置的 wenchain 网关，词级 ASR 走环境变量指定的子进程脚本。

四步（任何一步失败都返回 ok=False + 原因，不会抛栈）：
  1. 参考风格分析：抽帧 → VLM 逐帧读字幕 → 原生帧像素取色校准 → style_profile（带缓存）
  2. 念白逐字流：对**目标成片**跑词级 ASR → 逐字流 → 逐句念白
                 （给了 items 就用 items 的文本，更准；没给就用 ASR 识别结果）
  3. 目标编排：LLM 按参考风格把念白拆块、配色配位配特效、标关键词（+ 同音错字等长纠错）
  4. 烧录：生成 ASS，用 ffmpeg/libass 烧进视频

用法（库）:
    import sys; sys.path.insert(0, "/root/jmzhang/baidu/ViralForge/Agent_tools/caption_clone")
    from caption_clone import clone_captions
    r = clone_captions(video="/path/成片.mp4", ref_video="/path/参考爆款.mp4")
    # -> {"ok": True, "output": ".../成片_capfx.mp4", "blocks": 24, "lines": 12, ...}

用法（命令行）:
    python caption_clone.py --video 成片.mp4 --ref 参考爆款.mp4 [--out 输出.mp4]
    # stdout 打印一行 JSON；失败 exit code = 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import ass_burn, charstream, frames, profile_cache, target_match  # noqa: E402
from core._env import FFMPEG, has_libass  # noqa: E402


def _default_out(video: str) -> str:
    """输出路径缺省值：成片同名 + ``_capfx`` 后缀。"""
    base, ext = os.path.splitext(os.path.abspath(video))
    return base + "_capfx" + (ext or ".mp4")


def clone_captions(video: str, ref_video: str = "", out_video: str = "", items=None,
                   work_dir: str = "", typo_fix: bool = True, llm_model: str = "",
                   ref_fps: float = 0.0, mute_spans=None) -> dict:
    """给 video 烧上参考风格字幕，返回 dict（不抛异常）。

    Args:
        video:     目标成片（要烧字幕的那条视频）。
        ref_video: 参考视频（风格来源）。不给/文件不存在时用内置回退风格
                   （白字口播 + 红橙斜排大字），流程照跑。
        out_video: 输出路径，默认 `<成片名>_capfx.mp4`。
        items:     可选的字幕文本清单 `[{"start":秒,"end":秒,"text":"这一句念白"}]`。
                   有配音 plan 时**强烈建议给**：文本准，不会有同音错字；
                   不给就完全按成片 ASR 的识别结果做字幕。
        work_dir:  中间产物目录（抽帧/ASR/风格档/ASS/字幕清单.md），默认 `<成片同级>/<成片名>_capwork`。
        typo_fix:  同音错字纠错（只接受等长改写，不动时间轴）。
        llm_model: 覆盖拆块/纠错用的文本模型名。
        ref_fps:   参考视频抽帧率（默认 1.0 帧/秒，越大越准越慢越贵）。
        mute_spans: 禁烧区间 `[(起秒, 止秒)]` —— 画面自带字幕的窗口。这些窗口既不给 items，
                   也不让「补漏」把成片 ASR 的句子填回去，避免叠成双字幕。

    Returns:
        {"ok": True, "output": ..., "blocks": 块数, "lines": 句数, "work_dir": ...,
         "profile": {"density":..,"styles":[..]}, "cost_s": 秒}
        {"ok": False, "error": "原因", ...}
    """
    t0 = time.time()
    video = os.path.abspath(video or "")
    if not video or not os.path.isfile(video):
        return {"ok": False, "error": "成片不存在：{}".format(video)}
    ref_video = os.path.abspath(ref_video) if ref_video else ""
    if ref_video and not os.path.isfile(ref_video):
        print("[caption_clone] 参考视频不存在，改用内置回退风格：{}".format(ref_video), flush=True)
        ref_video = ""
    if not has_libass():
        return {"ok": False, "error": ("当前 ffmpeg 不含 libass，烧不了 ASS 字幕："
                                       "{}（用 FFMPEG 环境变量指一个带 libass 的）".format(FFMPEG))}
    out_video = os.path.abspath(out_video) if out_video else _default_out(video)
    work_dir = os.path.abspath(work_dir) if work_dir else os.path.splitext(video)[0] + "_capwork"
    os.makedirs(work_dir, exist_ok=True)
    model = llm_model or os.getenv("WHQ_CAPTION_LLM_MODEL") or None
    if typo_fix is False:
        os.environ["WHQ_CAPTION_TYPO_FIX"] = "0"

    try:
        profile = profile_cache.analyze_reference(
            ref_video, work_dir, fps=(ref_fps or None), model=None)
        stream = charstream.char_stream(video, os.path.join(work_dir, "final_asr"))
        if items:
            lines = charstream.build_lines(items, stream, mute_spans=mute_spans)
        else:
            print("[caption_clone] 没给 items，字幕文本用成片 ASR 识别结果", flush=True)
            lines = charstream.lines_from_stream(stream)
        if not lines:
            return {"ok": False, "error": ("拿不到任何念白句：成片没有人声，或词级 ASR 不可用"
                                           "（ASR_PYTHON/ASR_SCRIPT）且没有传 items"),
                    "work_dir": work_dir}
        lines = charstream.clamp_to(lines, frames.duration_seconds(video))
        if os.getenv("WHQ_CAPTION_TYPO_FIX", "1").strip().lower() not in (
                "0", "off", "false", "no"):
            target_match.correct_typos(lines, model=model)
        seq, by_role = target_match.plan(lines, profile, model=model)
        if not seq:
            return {"ok": False, "error": "LLM 没拆出任何字幕块", "work_dir": work_dir}
        md = os.path.join(work_dir, "目标字幕清单.md")
        target_match.dump_md(seq, by_role, md, os.path.basename(out_video),
                             os.path.basename(ref_video or "-"))
        done = ass_burn.burn(seq, video, out_video, work_dir)
        if not done:
            return {"ok": False, "error": "ASS 烧录失败（看上面的 ffmpeg 报错）",
                    "work_dir": work_dir}
        return {"ok": True, "output": done, "blocks": len(seq), "lines": len(lines),
                "work_dir": work_dir, "caption_md": md,
                "profile": {"density": profile.get("density"),
                            "styles": list(profile.get("styles", {}))},
                "cost_s": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001 —— 工具不能把调用方搞崩，一律转成 error
        return {"ok": False, "error": "{}: {}".format(type(exc).__name__, str(exc)[:300]),
                "work_dir": work_dir}


def main():
    """命令行入口：解析参数并调用 clone_captions()，stdout 打印一行 JSON，返回退出码。"""
    ap = argparse.ArgumentParser(description="给成片烧上和参考视频同风格的字幕")
    ap.add_argument("--video", required=True, help="目标成片")
    ap.add_argument("--ref", default="", help="参考视频（风格来源）；不给用内置回退风格")
    ap.add_argument("--out", default="", help="输出路径，默认 <成片名>_capfx.mp4")
    ap.add_argument("--items", default="", help='字幕文本清单 JSON 文件：[{"start","end","text"}]')
    ap.add_argument("--work-dir", default="", help="中间产物目录")
    ap.add_argument("--ref-fps", type=float, default=0.0, help="参考视频抽帧率，默认 1.0")
    ap.add_argument("--llm-model", default="", help="覆盖拆块用的文本模型名")
    ap.add_argument("--no-typo-fix", action="store_true", help="关掉同音错字纠错")
    args = ap.parse_args()
    items = None
    if args.items:
        with open(args.items, encoding="utf-8") as fh:
            items = json.load(fh)
    res = clone_captions(video=args.video, ref_video=args.ref, out_video=args.out,
                         items=items, work_dir=args.work_dir,
                         typo_fix=not args.no_typo_fix, llm_model=args.llm_model,
                         ref_fps=args.ref_fps)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
