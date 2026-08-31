# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""命令行入口：环境自检、建任务、跑流水线、看任务列表。

    viral_video_recreation doctor
    viral_video_recreation new --ref 参考片.mp4 --image 商品图.jpg --name 商品名
    viral_video_recreation run <task_id> [--from ingest]
    viral_video_recreation list

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import argparse
import os

__all__ = [
    'main',
]


def _doctor() -> int:
    """环境自检：配置、ffmpeg、产物目录都报一遍，缺什么直接说。"""
    from . import config, media
    print("产物目录：%s" % config.OUTPUT_DIR)
    print("配置文件：%s" % os.path.join(config.CONF_DIR, "config.env"))
    print("wenchain 网关：%s" % config.BASE_URL)
    print("理解链路：%s（gemini 网关 %d 个）"
          % (config.UNDERSTAND_ENGINE, len(config.GEMINI_GATEWAYS)))
    try:
        exe = media.ffmpeg()
        ok = media.run([exe, "-version"]).returncode == 0
        print("ffmpeg：%s（%s）" % (exe, "可用" if ok else "不可用"))
    except Exception as exc:   # noqa: BLE001  自检不该因为缺依赖而中断
        print("ffmpeg：不可用（%s）" % str(exc)[:200])
        return 1
    print("成片画布：%dx%d@%dfps" % (media.TARGET_W, media.TARGET_H, media.TARGET_FPS))
    missing = config.missing_keys()
    if missing:
        print("缺少配置（写进 conf/config.local.env 或 export）：%s" % "、".join(missing))
        return 1
    print("配置齐全")
    return 0


def _new(args) -> int:
    """建任务并登记素材。"""
    from . import task_store
    product = {"name": args.name, "category": args.category,
               "selling_points": args.point}
    options = {"copy_bgm": not args.no_bgm,
               "copy_subtitles": (True if args.subtitles
                                  else False if args.no_subtitles else None),
               "rewrite_mode": "creative" if args.creative else "strict"}
    rec = task_store.create_task(product, options,
                                 title=(args.name or "未命名") + " 复刻",
                                 slug=args.slug)
    task_id = rec["task_id"]
    task_store.register_input(task_id, "reference_video", args.ref)
    for path in args.image:
        task_store.register_input(task_id, "product_images", path)
    for path in args.video:
        task_store.register_input(task_id, "user_videos", path)
    if args.bgm:
        task_store.register_input(task_id, "bgm", args.bgm)
    print("任务已创建：%s" % task_id)
    print("任务目录：%s" % task_store.task_dir(task_id))
    return 0


def _run(args) -> int:
    """跑流水线，--from 给定时先重置该步骤及其后续。"""
    from . import pipeline
    if args.from_step:
        pipeline.reset_from(args.task_id, args.from_step)
    out = pipeline.run(args.task_id)
    print("状态：%s" % out["status"])
    if out.get("error"):
        print("失败原因：%s" % out["error"])
    return 0 if out["status"] == "completed" else 1


def _list(args) -> int:
    """打印任务摘要列表。"""
    from . import task_store
    rows = task_store.list_tasks(args.limit)
    if not rows:
        print("还没有任务")
        return 0
    for row in rows:
        print("%-40s %-10s %-12s %s" % (row["task_id"], row["status"] or "",
                                        row["step"] or "-", row["created_at"] or ""))
    return 0


def _parser() -> argparse.ArgumentParser:
    """组装子命令参数表。"""
    ap = argparse.ArgumentParser(prog="viral_video_recreation", description="通用爆款复刻")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="环境与配置自检")

    new = sub.add_parser("new", help="新建复刻任务并登记素材")
    new.add_argument("--ref", required=True, help="爆款参考视频")
    new.add_argument("--image", action="append", default=[], help="商品图，可重复")
    new.add_argument("--video", action="append", default=[], help="用户素材视频，可重复")
    new.add_argument("--bgm", default="", help="指定 BGM 音频")
    new.add_argument("--name", default="", help="商品名")
    new.add_argument("--category", default="", help="品类")
    new.add_argument("--point", action="append", default=[], help="卖点，可重复")
    new.add_argument("--slug", default="", help="任务目录名，留空用时间戳")
    new.add_argument("--no-bgm", action="store_true", help="不复刻参考片音乐")
    new.add_argument("--subtitles", action="store_true", help="强制烧字幕")
    new.add_argument("--no-subtitles", action="store_true", help="强制不烧字幕")
    new.add_argument("--creative", action="store_true", help="允许改写剧本")

    run = sub.add_parser("run", help="跑流水线（已完成的步骤自动跳过）")
    run.add_argument("task_id")
    run.add_argument("--from", dest="from_step", default="",
                     help="从这一步开始重跑，旧产物留档 history/")

    ls = sub.add_parser("list", help="任务列表")
    ls.add_argument("--limit", type=int, default=50)
    return ap


def main(args=None):
    """主程序入口，返回 0 表示成功。"""
    if args is None:
        import sys
        args = sys.argv[1:]
    parsed = _parser().parse_args(args)
    if parsed.cmd == "doctor":
        return _doctor()
    if parsed.cmd == "new":
        return _new(parsed)
    if parsed.cmd == "run":
        return _run(parsed)
    return _list(parsed)
