"""ffmpeg / ffprobe 薄封装：探时长、裁片段、抽音轨、抽帧。

裁片段带「放慢补时长」逻辑（用户素材比分镜短时最多放慢 MAX_SLOWDOWN 倍）。
"""
import os

import analyze_materials  # pyright: ignore[reportImplicitRelativeImport]
import produce_video  # pyright: ignore[reportImplicitRelativeImport]

MAX_SLOWDOWN = 1.6           # 用户素材比分镜短时最多放慢多少倍来补足时长


# ---------------- 媒体工具 ----------------
def _ffmpeg() -> str:
    return produce_video._ffmpeg()


def _probe(path: str) -> dict:
    err = produce_video._run([_ffmpeg(), "-hide_banner", "-i", path]).stderr
    return {"file": path, "name": os.path.basename(path),
            "size_mb": round(os.path.getsize(path) / (1 << 20), 2),
            "duration_sec": round(produce_video._duration(path), 1),
            "has_audio": "Audio:" in err}


def _sec(value) -> float:
    return max(0.0, analyze_materials._sec(value))


def _yn(value) -> str:
    """布尔事实写进日志：None 是「没判出来」，和 False 必须区分开。"""
    return {True: "有", False: "无"}.get(value, "未判定")


def cut_clip(src: str, start: float, avail: float, need: float, dst: str) -> dict:
    """从用户素材里裁一段并对齐到分镜时长。

    素材比分镜短时用 setpts/atempo 放慢补足（最多 MAX_SLOWDOWN 倍），
    再长就直接截断；输出统一到成片分辨率与帧率，方便后面 concat。
    """
    take = max(0.4, min(avail, need))
    speed = 1.0
    if need > take * 1.02:
        speed = max(1.0 / MAX_SLOWDOWN, take / need)   # <1 表示放慢
    # setsar=1 不能省：scale 默认保持显示宽高比，会把输入的 SAR（非方形像素）原样传下去，
    # 于是「已统一分辨率」的产物里 SAR 各不相同。segment_build._concat_same 是 -c copy 直接
    # 拼流，ffmpeg 不报错也不告警，成片文件头只写第一段的 SAR，后面几段画面被按错误比例
    # 拉伸。scale+pad 算黑边几何时本来也假设了方形像素，SAR≠1 的素材会被垫歪。
    vf = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=%d"
          % (produce_video.TARGET_W, produce_video.TARGET_H, produce_video.TARGET_W,
             produce_video.TARGET_H, produce_video.TARGET_FPS))
    if speed < 1.0:
        vf = "setpts=%.4f*PTS," % (1.0 / speed) + vf
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.2f" % start, "-t", "%.2f" % take, "-i", src]
    probe = _probe(src)
    if not probe["has_audio"]:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest"]
    cmd += ["-vf", vf]
    if probe["has_audio"] and speed < 1.0:
        cmd += ["-filter:a", "atempo=%.4f" % max(0.5, speed)]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += produce_video.aac_args() + [dst]
    ret = produce_video._run(cmd)
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("裁剪素材失败：%s" % ret.stderr[-300:])
    return {"file": dst, "start_sec": round(start, 2), "take_sec": round(take, 2),
            "speed": round(speed, 3), "duration_sec": round(produce_video._duration(dst), 2)}


def extract_audio(video: str, dst: str) -> str:
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-i", video, "-vn", "-c:a", "aac", "-b:a", "160k", dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("提取音轨失败：%s" % ret.stderr[-300:])
    return dst


def extract_frame(video: str, sec: float, dst: str, width: int = 0) -> str:
    """抽一帧存成 jpg。sec 是相对整条视频的绝对秒数。

    width 大于 0 时按宽度等比缩小，给列表缩略图用；默认 0 保持原分辨率。
    """
    scale = ["-vf", "scale=%d:-2" % width] if width > 0 else []
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-ss", "%.2f" % max(0.0, sec), "-i", video,
                             "-frames:v", "1", "-q:v", "3"] + scale + [dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("抽帧失败：%s" % ret.stderr[-300:])
    return dst
