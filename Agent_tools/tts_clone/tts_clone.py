#!/usr/bin/env python3
"""tts_clone —— zero-shot 声音克隆工具（给一段参考人声 + 一段文案，产出用该音色念出的 wav）。

从 Viral_Video_Agent 的 `src/editing/tts.py` 抽出来的独立版本：去掉了项目内部依赖
（obs 日志、imageio_ffmpeg、Split 仓 .env），只依赖 **python3 标准库 + ffmpeg**。
真正跑模型的是子进程里的后端脚本（默认 backends/run_voxcpm2_zero_shot.py，VoxCPM2），
它需要自己的 conda env；本文件只负责「抽参考音 → 调后端 → 响度归一 → 报时长」。

用法（库）:
    from tts_clone import clone
    r = clone(ref_audio="/path/a.mov", ref_text="参考音里说的原话",
              text="要念出来的文案", out_wav="/tmp/out.wav", ref_range="12.0-18.0")
    # -> {"ok": True, "output": "/tmp/out.wav", "duration": 3.42, "lufs_normalized": True}

用法（命令行）:
    python tts_clone.py --ref a.mov --ref-range 12.0-18.0 \
        --ref-text "参考音里说的原话" --text "要念出来的文案" --out out.wav
    # stdout 打印一行 JSON（同上），失败时 exit code = 1

后端契约（换模型只要满足这个 CLI 即可）:
    <python> <script> --prompt-wav P.wav --prompt-asr P.json --text "文案" --output OUT.wav
    P.json = {"results": [{"text": "参考音转写"}]}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))
_log = logging.getLogger("tts_clone")


def _first_exec(*candidates):
    """返回第一个可执行的候选（PATH 里的名字或绝对路径）；都不行就返回第一个，让报错带上原路径。"""
    for cand in candidates:
        if os.path.isabs(cand):
            if os.access(cand, os.X_OK):
                return cand
        else:
            from shutil import which
            if which(cand):
                return cand
    return candidates[0]


# ffmpeg：必须能用（抽参考音 + 响度归一都靠它）。系统 PATH 没有就按候选顺序找，FFMPEG 可覆盖。
FFMPEG = os.getenv("FFMPEG") or _first_exec(
    "ffmpeg",
    "/root/miniconda3/envs/voxcpm/bin/ffmpeg",
    "/root/miniconda3/envs/media/bin/ffmpeg",
)
# 跑模型的解释器/脚本。默认指向本目录自带的 VoxCPM2 后端 + 它的 conda env
# （TTS_CLONE_PYTHON 覆盖；权重位置由后端脚本的 MODEL_ROOT/VOXCPM_MODEL_DIR 决定）。
TTS_PYTHON = os.getenv("TTS_CLONE_PYTHON", "/root/miniconda3/envs/voxcpm/bin/python")
TTS_SCRIPT = os.getenv("TTS_CLONE_SCRIPT",
                       os.path.join(_HERE, "backends", "run_voxcpm2_zero_shot.py"))
# 单次合成的超时（秒）：首次调用要加载模型权重，给足时间
TIMEOUT = int(os.getenv("TTS_CLONE_TIMEOUT", "600"))
# 产出响度目标（LUFS）。"off" 关闭归一，保留模型原始电平。
LUFS = os.getenv("TTS_CLONE_LUFS", "-16")
# 参考音采样率：**不要改成 44.1k**。VoxCPM2 的 AudioVAE 编码率就是 16000，
# 48k 输出是生成式扩带宽、不吃参考音高频；实测喂 44.1k 产出反而更闷
# （>8kHz 能量占比 7.6% vs 16k 的 15.5%，多一道重采样滚降更狠）。
PROMPT_SR = int(os.getenv("TTS_CLONE_PROMPT_SR", "16000"))
# 参考音转写要不要喂给模型：
#   basic（默认）—— 只给参考音波形。产出**就是目标文案**，不多不少。
#   ultimate     —— 参考音 + 其转写一起给（VoxCPM2 的 ultimate cloning），音色理论上更贴，
#                   但实测**输出不可控**：2/2 次把参考音的原话也念了出来（14 字文案产出
#                   5.76s/7.84s，ASR 回听是「参考原话 + 目标文案」），或把目标文案开头
#                   吞掉改写（29 字文案只念出后 21 字）。basic 模式同样两条用例都干净。
#                   所以默认 basic，要试 ultimate 请自己 ASR 回听校验。
PROMPT_MODE = os.getenv("TTS_CLONE_PROMPT_MODE", "basic")


def _parse_range(text: str) -> tuple:
    """"12.5-18.0" -> (12.5, 18.0)；给不出就 (0, 0)（= 用整段素材做参考）。"""
    try:
        a, b = str(text).split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _extract_prompt_wav(src: str, time_range: str, out_wav: str) -> bool:
    """从参考素材（视频/音频均可）抽一段 16k 单声道 wav 作克隆的 prompt。"""
    start, end = _parse_range(time_range)
    dur = max(0.5, end - start) if end > start else 0.0
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "{:.3f}".format(max(0.0, start)), "-i", src]
    if dur > 0:
        cmd += ["-t", "{:.3f}".format(dur)]
    cmd += ["-vn", "-ac", "1", "-ar", str(PROMPT_SR), out_wav]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if r.returncode != 0:
            _log.warning("参考音抽取失败：%s", r.stderr.decode("utf-8", "ignore")[-200:])
            return False
        return os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0
    except (subprocess.SubprocessError, OSError) as exc:
        _log.warning("参考音抽取异常：%s", exc)
        return False


def _normalize_loudness(path: str, target: str = None) -> bool:
    """把产出 wav 拉到统一口播响度（LUFS，默认 -16）。两遍 loudnorm：先测量再按测量值套用。

    zero-shot 克隆会把参考音的**响度**一起克隆，所以响度不能交给素材决定：实测参考素材本身
    mean -39dB 时，产出的配音全在 -40dB 左右，混上 BGM 后几乎听不见人声。
    """
    target = str(LUFS if target is None else target).strip()
    if target.lower() in ("off", "no", "false", ""):
        return False
    try:
        probe = subprocess.run(
            [FFMPEG, "-hide_banner", "-nostats", "-i", path,
             "-af", "loudnorm=I={}:TP=-1.5:LRA=11:print_format=json".format(target),
             "-f", "null", os.devnull],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        text = probe.stderr.decode("utf-8", "ignore")
        head = text.rfind("{")
        meas = None
        while head >= 0 and meas is None:
            try:
                meas = json.loads(text[head:text.rindex("}") + 1])
            except ValueError:
                head = text.rfind("{", 0, head)
        if not (meas and "input_i" in meas):
            _log.warning("响度归一跳过（loudnorm 没给测量值）")
            return False
        flt = ("loudnorm=I={}:TP=-1.5:LRA=11:measured_I={}:measured_TP={}:measured_LRA={}"
               ":measured_thresh={}:linear=true".format(
                   target, meas["input_i"], meas["input_tp"], meas["input_lra"],
                   meas["input_thresh"]))
        tmp = path + ".norm.wav"
        r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                            "-af", flt, "-ar", "44100", tmp],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if r.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            _log.warning("响度归一失败（保留原始电平）：%s",
                         r.stderr.decode("utf-8", "ignore")[-160:])
            return False
        os.replace(tmp, path)
        _log.info("响度归一：%.1f LUFS -> %s LUFS", float(meas["input_i"]), target)
        return True
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
        _log.warning("响度归一异常（保留原始电平）：%s", str(exc)[:160])
        return False


def _wav_duration(path: str) -> float:
    """读 wav 时长：先用 stdlib wave（PCM），失败再手解析 RIFF 头（兼容 float wav）。"""
    try:
        with wave.open(path, "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
            if rate:
                return round(frames / float(rate), 2)
    except (wave.Error, OSError):
        pass
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return 0.0
        pos, rate, ch, bits, data_bytes = 12, 0, 1, 16, 0
        while pos + 8 <= len(data):
            cid = data[pos:pos + 4]
            size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            body = data[pos + 8:pos + 8 + size]
            if cid == b"fmt " and len(body) >= 16:
                ch = struct.unpack("<H", body[2:4])[0] or 1
                rate = struct.unpack("<I", body[4:8])[0]
                bits = struct.unpack("<H", body[14:16])[0] or 16
            elif cid == b"data":
                data_bytes = size
            pos += 8 + size + (size & 1)
        if rate and ch and bits:
            return round(data_bytes / float(rate * ch * (bits // 8)), 2)
    except (OSError, struct.error):
        pass
    return 0.0


def available() -> bool:
    """后端解释器/脚本是否就绪（不就绪时 clone 会返回明确的 error，不会崩）。"""
    return bool(TTS_PYTHON and os.path.exists(TTS_PYTHON)
                and TTS_SCRIPT and os.path.exists(TTS_SCRIPT))


def clone(ref_audio: str, ref_text: str, text: str, out_wav: str,
          ref_range: str = "", lufs: str = None, prompt_mode: str = None) -> dict:
    """用 ref_audio 里的音色把 text 念出来，写到 out_wav。

    Args:
        ref_audio: 参考人声文件（mp4/mov/wav/m4a… 有音轨就行）。
        ref_text:  参考音里**实际说的原话**。只有 prompt_mode="ultimate" 时会喂给模型；
                   默认的 basic 模式忽略它（留着是为了两种模式共用同一套调用参数）。
        text:      要合成的目标文案。
        out_wav:   输出 wav 路径（父目录会自动建）。
        ref_range: 可选 "起-止"（秒），只取参考音的这一段；不给则用整段。
        lufs:      可选，覆盖本次的响度目标（"-16"/"off"）。
        prompt_mode: 可选，"basic"（默认，产出只含目标文案）或 "ultimate"（连参考转写
                   一起喂，音色更贴但实测会多念参考原话/吞掉文案开头）。

    Returns:
        {"ok": True, "output": ..., "duration": 秒, "lufs_normalized": bool}
        {"ok": False, "error": "原因"}
    """
    text = (text or "").strip()
    ref_text = (ref_text or "").strip()
    mode = (prompt_mode or PROMPT_MODE or "basic").strip().lower()
    if not text:
        return {"ok": False, "error": "缺少要合成的文案 text"}
    if mode == "ultimate" and not ref_text:
        return {"ok": False, "error": "ultimate 模式需要参考音转写 ref_text（或改用 basic 模式）"}
    if not ref_audio or not os.path.isfile(ref_audio):
        return {"ok": False, "error": "参考素材不存在：{}".format(ref_audio)}
    if not available():
        return {"ok": False, "error": ("后端未就绪：TTS_CLONE_PYTHON={} TTS_CLONE_SCRIPT={}"
                                       .format(TTS_PYTHON, TTS_SCRIPT))}
    out_wav = os.path.abspath(out_wav)
    tmpdir = tempfile.mkdtemp(prefix="ttsclone_")
    prompt_wav = os.path.join(tmpdir, "prompt.wav")
    prompt_asr = os.path.join(tmpdir, "prompt_asr.json")
    try:
        if not _extract_prompt_wav(ref_audio, ref_range, prompt_wav):
            return {"ok": False, "error": "参考音抽取失败（检查 ffmpeg 与该文件是否有音轨）"}
        with open(prompt_asr, "w", encoding="utf-8") as fh:
            # basic 模式给空转写：后端据此退化为「只按参考音波形克隆」，产出就是目标文案。
            json.dump({"results": [{"text": ref_text if mode == "ultimate" else ""}]},
                      fh, ensure_ascii=False)
        os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
        env = dict(os.environ)
        # 清掉会污染后端解释器导入的变量：否则子进程串到调用方的 site-packages，
        # 后端 env 里的模型包会报 ModuleNotFoundError。
        for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
            env.pop(var, None)
        cmd = [TTS_PYTHON, TTS_SCRIPT, "--prompt-wav", prompt_wav, "--prompt-asr", prompt_asr,
               "--text", text, "--output", out_wav]
        _log.info("clone(%s): text=%r ref=%s", os.path.basename(TTS_SCRIPT),
                  text[:40], os.path.basename(ref_audio))
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired:
            # 契约是「不抛栈」：超时也要返回 error 让调用方决定跳过还是重试。
            # CPU 上并发跑多条最容易撞这里（单条约 130s，三条并发就会超 600s）。
            return {"ok": False, "error": "声音克隆超时（{}s，TTS_CLONE_TIMEOUT 可调）"
                                          .format(TIMEOUT)}
        if r.returncode != 0:
            return {"ok": False, "error": "声音克隆失败（{}）：{}".format(
                os.path.basename(TTS_SCRIPT), r.stderr.decode("utf-8", "ignore")[-400:])}
        if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
            return {"ok": False, "error": "声音克隆未产出音频"}
        normalized = _normalize_loudness(out_wav, lufs)
        return {"ok": True, "output": out_wav, "duration": _wav_duration(out_wav),
                "lufs_normalized": normalized}
    finally:
        for p in (prompt_wav, prompt_asr):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def main():
    """命令行入口：--ref 参考人声 + --text 文案 → --out 产出 wav，结果以一行 JSON 打印。"""
    ap = argparse.ArgumentParser(description="zero-shot 声音克隆：参考人声 + 文案 -> wav")
    ap.add_argument("--ref", required=True, help="参考人声文件（视频/音频）")
    ap.add_argument("--ref-text", default="", help="参考音里实际说的原话（仅 ultimate 模式用到）")
    ap.add_argument("--text", help="要合成的文案（与 --text-file 二选一）")
    ap.add_argument("--text-file", help="要合成的文案（从文件读，避免 shell 转义）")
    ap.add_argument("--out", required=True, help="输出 wav 路径")
    ap.add_argument("--ref-range", default="", help='可选，只取参考音的这一段，如 "12.0-18.0"')
    ap.add_argument("--lufs", default=None, help='响度目标，默认 -16；"off" 关闭归一')
    ap.add_argument("--prompt-mode", default=None, choices=["basic", "ultimate"],
                    help="basic(默认)=只给参考音波形；ultimate=连转写一起给(输出可能多念参考原话)")
    ap.add_argument("--quiet", action="store_true", help="只打印结果 JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="[tts_clone] %(message)s")
    text = args.text or ""
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read().strip()
    res = clone(ref_audio=args.ref, ref_text=args.ref_text, text=text,
                out_wav=args.out, ref_range=args.ref_range, lufs=args.lufs,
                prompt_mode=args.prompt_mode)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
