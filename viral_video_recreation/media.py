# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""ffmpeg / ffprobe 薄封装：探时长、归一化、拼接、裁片段、抽音轨、抽帧。

ffmpeg 二进制取自 imageio_ffmpeg，不依赖系统安装。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import os
import subprocess

from . import config

MAX_SLOWDOWN = 1.6      # 用户素材比分镜短时最多放慢多少倍来补足时长
TARGET_FPS = 30
# 音轨参数必须统一：concat 是流复制，一条流里声道数中途从 mono 变 stereo 时，
# 后面再重编码这条流，AAC 编码器会在切换点吐 NaN/Inf 直接失败。
TARGET_AR, TARGET_AC = 44100, 2


def canvas() -> "tuple[int, int]":
    """归一化画布跟随 config.ASPECT（生成画幅），短边 720，宽高取偶。

    写死 9:16 会把横屏参考片的复刻成片垫成上下黑边。
    """
    try:
        width, height = (int(x) for x in str(config.ASPECT).split(":"))
    except (ValueError, AttributeError):
        width, height = 9, 16
    short = 720
    if width >= height:
        return max(2, round(short * width / height / 2) * 2), short
    return short, max(2, round(short * height / width / 2) * 2)


TARGET_W, TARGET_H = canvas()


def aac_args(bitrate: str = "128k") -> list:
    """所有会进 concat 的片段都用这套音轨参数编码，见 TARGET_AR/TARGET_AC 的说明。"""
    return ["-c:a", "aac", "-b:a", bitrate, "-ar", str(TARGET_AR), "-ac", str(TARGET_AC)]


def ffmpeg() -> str:
    """imageio_ffmpeg 自带的 ffmpeg 二进制，避免依赖系统安装。"""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list) -> subprocess.CompletedProcess:
    """跑一条命令并把 stdout/stderr 收回来，失败不抛异常，由调用方判 returncode。"""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _scale_pad() -> str:
    """归一化用的 filter：等比缩放后补黑边到目标画布，并统一帧率。"""
    return ("scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=%d"
            % (TARGET_W, TARGET_H, TARGET_W, TARGET_H, TARGET_FPS))


def duration(path: str) -> float:
    """从 ffmpeg 输出里读时长，取不到返回 0。"""
    for line in run([ffmpeg(), "-hide_banner", "-i", path]).stderr.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            try:
                hour, minute, sec = hms.split(":")
                return int(hour) * 3600 + int(minute) * 60 + float(sec)
            except ValueError:
                return 0.0
    return 0.0


def has_audio(path: str) -> bool:
    """有没有音轨。None 和 False 语义不同，这里只回答确定有/没有。"""
    return "Audio:" in run([ffmpeg(), "-hide_banner", "-i", path]).stderr


def probe(path: str) -> dict:
    """媒体基础事实：文件名、大小、时长、有无音轨。"""
    err = run([ffmpeg(), "-hide_banner", "-i", path]).stderr
    return {"file": path, "name": os.path.basename(path),
            "size_mb": round(os.path.getsize(path) / (1 << 20), 2),
            "duration_sec": round(duration(path), 1),
            "has_audio": "Audio:" in err}


def cut_clip(src: str, start: float, avail: float, need: float, dst: str) -> dict:
    """从素材里裁一段并对齐到分镜时长。

    素材比分镜短时用 setpts/atempo 放慢补足（最多 MAX_SLOWDOWN 倍），再长就直接截断；
    输出统一到成片分辨率与帧率，方便后面 concat。
    """
    take = max(0.4, min(avail, need))
    speed = 1.0
    if need > take * 1.02:
        speed = max(1.0 / MAX_SLOWDOWN, take / need)    # <1 表示放慢
    video_filter = _scale_pad()
    if speed < 1.0:
        video_filter = "setpts=%.4f*PTS," % (1.0 / speed) + video_filter
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.2f" % start, "-t", "%.2f" % take, "-i", src]
    info = probe(src)
    if not info["has_audio"]:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=%d" % TARGET_AR, "-shortest"]
    cmd += ["-vf", video_filter]
    if info["has_audio"] and speed < 1.0:
        cmd += ["-filter:a", "atempo=%.4f" % max(0.5, speed)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += aac_args() + [dst]
    ret = run(cmd)
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("裁剪素材失败：%s" % ret.stderr[-300:])
    return {"file": dst, "start_sec": round(start, 2), "take_sec": round(take, 2),
            "speed": round(speed, 3), "duration_sec": round(duration(dst), 2)}


def extract_audio(video: str, dst: str) -> str:
    """抽出整条音轨存成 aac。"""
    ret = run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", video,
               "-vn", "-c:a", "aac", "-b:a", "160k", dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("提取音轨失败：%s" % ret.stderr[-300:])
    return dst


def extract_frame(video: str, sec: float, dst: str) -> str:
    """抽一帧存成 jpg。sec 是相对整条视频的绝对秒数。"""
    ret = run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
               "-ss", "%.2f" % max(0.0, sec), "-i", video,
               "-frames:v", "1", "-q:v", "2", dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("抽帧失败：%s" % ret.stderr[-300:])
    return dst


def normalize(src: str, dst: str) -> bool:
    """统一分辨率/帧率/编码，并保证一定有音轨（没有就补静音），concat 才能直接复制流。"""
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    if not has_audio(src):
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=%d" % TARGET_AR, "-shortest"]
    cmd += ["-vf", _scale_pad(), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"] + aac_args() + [dst]
    ret = run(cmd)
    return ret.returncode == 0 and os.path.isfile(dst)


def concat(files: list, outdir: str, name: str = "final.mp4") -> dict:
    """把各段拼成成片。返回 {"file", "segments_used"} 或 {"error"}。

    有片段归一化失败就整体报错：少一段的成片是静默内容丢失，比拼接失败更难发现。
    """
    if not files:
        return {"error": "没有可拼接的片段"}
    norm_dir = os.path.join(outdir, "normalized")
    os.makedirs(norm_dir, exist_ok=True)
    normed, bad = [], []
    for i, src in enumerate(files):
        dst = os.path.join(norm_dir, "n%02d.mp4" % i)
        if normalize(src, dst):
            normed.append(dst)
        else:
            bad.append(os.path.basename(src))
    if bad:
        return {"error": "这些片段归一化失败，拼接中止（少一段的成片没有意义）：%s"
                         % "、".join(bad)}
    listfile = os.path.join(norm_dir, "list.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for path in normed:
            fh.write("file '%s'\n" % path.replace("'", "'\\''"))
    final = os.path.join(outdir, name)
    ret = run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
               "-safe", "0", "-i", listfile, "-c", "copy", "-movflags", "+faststart",
               final])
    if ret.returncode != 0 or not os.path.isfile(final):
        return {"error": "concat 失败: %s" % ret.stderr[-300:]}
    return {"file": final, "segments_used": len(normed)}
