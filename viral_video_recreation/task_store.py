# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""任务目录与素材登记：一个任务 = <OUTPUT_DIR>/tasks/{task_id}/。

参数、中间产物、状态、日志全部落在任务目录里，所有步骤都通过这里读写任务记录。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import glob
import json
import os
import threading
import time
import uuid

from . import config

# 并行阶段多线程共享同一个任务 rec：写状态、追日志、落盘 task.json 都必须串行，
# 否则 json.dump 迭代到一半 dict 变了会直接炸。
_LOCK = threading.RLock()

DEFAULT_OPTIONS = {
    "copy_bgm": True,
    "copy_reference_audio": False,
    "copy_subtitles": None,        # None=跟随参考片：参考片有字幕才烧
    "rewrite_mode": "strict",      # strict | creative
    "use_user_materials": True,
    "seg_workers": 5,
}

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
KIND_EXTS = {"reference_video": VIDEO_EXTS, "user_videos": VIDEO_EXTS,
             "product_images": IMAGE_EXTS, "person_images": IMAGE_EXTS,
             "bgm": AUDIO_EXTS}


def tasks_dir() -> str:
    """任务根目录。每次读 config.OUTPUT_DIR，测试里改环境变量后无需重载模块。"""
    return os.path.join(config.OUTPUT_DIR, "tasks")


def task_dir(task_id: str) -> str:
    """单个任务的目录。"""
    return os.path.join(tasks_dir(), task_id)


def task_path(task_id: str, *parts: str) -> str:
    """任务目录下的文件路径，父目录自动创建。"""
    path = os.path.join(task_dir(task_id), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def task_subdir(task_id: str, *parts: str) -> str:
    """任务目录下的子目录，自动创建。"""
    path = os.path.join(task_dir(task_id), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def rel(task_id: str, path: str) -> str:
    """产物路径转成相对任务目录的写法，落进 task.json 里更短也便于搬迁。"""
    return os.path.relpath(path, task_dir(task_id))


def load(task_id: str) -> dict:
    """读任务记录。"""
    with open(os.path.join(task_dir(task_id), "task.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(rec: dict) -> dict:
    """任务记录落盘。先写 .tmp 再 rename，避免轮询方读到写一半的 json。"""
    with _LOCK:
        rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = task_path(rec["task_id"], "task.json")
        tmp = path + ".tmp"        # 前端在轮询，写一半的 json 不能被读到
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return rec


def log(rec: dict, msg: str) -> None:
    """追一条任务日志，同时打到 stdout。只留最近 400 条。"""
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    with _LOCK:
        rec.setdefault("logs", []).append(line)
        rec["logs"] = rec["logs"][-400:]
    print("[%s] %s" % (rec["task_id"], msg), flush=True)


def safe_slug(text: str) -> str:
    """目录名安全化：保留中英文与数字，其余字符（路径分隔符、空格、标点）换成下划线。"""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:80]


def create_task(product: dict = None, options: dict = None, title: str = "",
                slug: str = "") -> dict:
    """建任务并落盘。product 可以先留空，上传素材后再补。

    slug 给定时用它当 task_id（目录名可读，如 20260831_参考A生成B），重名自动追加
    _2/_3；不给则退回「时间戳-哈希」格式。
    """
    if slug and safe_slug(slug):
        base = safe_slug(slug)
        task_id, seq = base, 2
        while os.path.isdir(task_dir(task_id)):
            task_id = "%s_%d" % (base, seq)
            seq += 1
    else:
        task_id = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:4])
    opts = dict(DEFAULT_OPTIONS)
    opts.update({k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS})
    rec = {
        "task_id": task_id,
        "title": title or "未命名复刻任务",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "draft",
        "step": "",
        "options": opts,
        "product": {"name": "", "selling_points": [], "category": "", "description": ""},
        "inputs": {"reference_video": "", "product_images": [], "user_videos": [],
                   "person_images": [], "bgm": ""},
        "steps": {},
        "logs": [],
    }
    rec["product"].update({k: v for k, v in (product or {}).items()
                           if k in rec["product"]})
    os.makedirs(task_path(task_id, "assets", "uploads"), exist_ok=True)
    log(rec, "任务已创建")
    return save(rec)


def update_task(task_id: str, product: dict = None, options: dict = None,
                title: str = None) -> dict:
    """改商品信息 / 复刻参数 / 标题，未给的字段保持不变。"""
    rec = load(task_id)
    if product:
        rec["product"].update({k: v for k, v in product.items() if k in rec["product"]})
    if options:
        rec["options"].update({k: v for k, v in options.items() if k in DEFAULT_OPTIONS})
    if title:
        rec["title"] = title
    return save(rec)


def list_tasks(limit: int = 50) -> list:
    """任务摘要列表，最近的在前。"""
    out = []
    pattern = os.path.join(tasks_dir(), "*", "task.json")
    for path in sorted(glob.glob(pattern), reverse=True)[:limit]:
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({k: rec.get(k)
                    for k in ("task_id", "title", "status", "step", "created_at")})
    return out


def register_input(task_id: str, kind: str, path: str) -> dict:
    """把素材文件登记进 inputs。kind 见 KIND_EXTS。"""
    if kind not in KIND_EXTS:
        raise ValueError("不支持的素材类型：%s" % kind)
    if not os.path.isfile(path):
        raise ValueError("文件不存在：%s" % path)
    if not path.lower().endswith(KIND_EXTS[kind]):
        raise ValueError("%s 只接受 %s" % (kind, "/".join(KIND_EXTS[kind])))
    path = os.path.abspath(path)
    rec = load(task_id)
    rec["inputs"].setdefault(kind, "" if kind == "bgm" else [])   # 老任务缺新输入位
    if isinstance(rec["inputs"][kind], list):
        if path not in rec["inputs"][kind]:
            rec["inputs"][kind].append(path)
    else:
        rec["inputs"][kind] = path
    log(rec, "登记素材 %s：%s" % (kind, os.path.basename(path)))
    return save(rec)


def remove_input(task_id: str, kind: str, path: str) -> dict:
    """把一个素材从 inputs 里摘掉，文件本身不删。"""
    rec = load(task_id)
    path = os.path.abspath(path)
    if isinstance(rec["inputs"].get(kind), list):
        rec["inputs"][kind] = [p for p in rec["inputs"][kind] if p != path]
    elif rec["inputs"].get(kind) == path:
        rec["inputs"][kind] = ""
    return save(rec)
