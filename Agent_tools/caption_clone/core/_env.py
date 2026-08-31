"""_env —— 替代原 whq_clone/_common：只解析本工具需要的外部可执行文件路径。

原包用 `_common` 挂 sys.path + 解析 REPO/FFMPEG。抽成独立工具后不需要挂路径，只需要：
- FFMPEG：**必须同时带 libass（烧字幕）和 libx264（重编码输出）**。系统 ffmpeg 常缺 libass，
  conda 里的 `media` 这类 `--disable-gpl` 构建带 libass 但没有 libx264，一样烧不出来。
  所以候选按「两样都有」筛，用 FFMPEG 环境变量可强制指定。
- FFPROBE：由 FFMPEG 推导，也可用 FFPROBE 覆盖。
"""
import os
import shutil
import subprocess

# 候选顺序：环境变量 > PATH > 本机已知 libass + libx264 都有的 conda 环境。
# 注意 media 环境是 --disable-gpl（无 libx264），故意不放进来。
_FFMPEG_CANDIDATES = (
    "/root/miniconda3/envs/voxcpm/bin/ffmpeg",
    "/root/miniconda3/envs/cutclaw/bin/ffmpeg",
    "/root/miniconda3/envs/liveclip_vocal/bin/ffmpeg",
    "/root/miniconda3/envs/storyline/bin/ffmpeg",
)


def _probe(ffmpeg, flag):
    """跑 ``ffmpeg -hide_banner <flag>`` 拿 stdout（用于查 filters/encoders），失败返回空串。"""
    try:
        return subprocess.run([ffmpeg, "-hide_banner", flag],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=30).stdout.decode("utf-8", "ignore")
    except (OSError, subprocess.SubprocessError):
        return ""


def _usable(ffmpeg):
    """libass（subtitles/ass 滤镜）+ libx264（重编码）都在才算可用。"""
    filters = _probe(ffmpeg, "-filters")
    if " ass " not in filters and " subtitles " not in filters:
        return False
    return " libx264 " in _probe(ffmpeg, "-encoders")


def _pick_ffmpeg():
    """按「环境变量 FFMPEG > PATH > 预置 conda 候选」的顺序选一个可用的 ffmpeg 路径。"""
    env = os.getenv("FFMPEG")
    if env:
        return env
    found = shutil.which("ffmpeg")
    if found and _usable(found):
        return found
    for cand in _FFMPEG_CANDIDATES:
        if os.access(cand, os.X_OK) and _usable(cand):
            return cand
    return found or "ffmpeg"


FFMPEG = _pick_ffmpeg()
FFPROBE = os.getenv("FFPROBE") or (
    FFMPEG.replace("ffmpeg", "ffprobe") if FFMPEG.endswith("ffmpeg") else "ffprobe")


def has_libass():
    """当前 FFMPEG 是否带 libass（不带则 ass 烧录必失败，提前给出明确原因）。"""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=30).stdout.decode("utf-8", "ignore")
    except (OSError, subprocess.SubprocessError):
        return False
    return " ass " in out or "\nass " in out or " subtitles " in out
