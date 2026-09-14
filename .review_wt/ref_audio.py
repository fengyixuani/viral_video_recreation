"""参考片音轨判定与分离（步骤 audio）。

判参考片音轨里有什么（口播 / 纯音乐 / 都有），按 rules 的「参考片音轨」表决定
复刻 BGM 时用整轨还是 demucs 分离出的伴奏，顺带产出纯人声供音色克隆用。
"""
import json
import os
import shutil
import subprocess

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import gates  # pyright: ignore[reportImplicitRelativeImport]
import produce_video  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from media import _ffmpeg, _yn, extract_audio
from task_store import _p, _rel, log


# ---------------- 步骤 3：参考片音频判定 ----------------
# 判定口径见 rules.py 的「参考片音轨」表；这里只负责采集事实 + 执行动作。
# 事实优先用 analyze_reference 拆解时直接给出的「有背景音乐 / 有人声口播」，缺了才补问音频模型。
AUDIO_CLASS_PROMPT = """听这段音频，只依据实际听到的内容判断，输出 json：
{"有背景音乐": true/false,
 "有人声口播": true/false,
 "音乐说明": "有音乐就描述风格与出现位置，没有就写「无」",
 "人声说明": "有人声就说明是台词/旁白/口播，没有就写「无」",
 "置信度": 0-1 的小数}
判断标准：背景音乐指持续的旋律或节奏配乐，环境音、音效不算音乐；
人声口播指能听出说话内容的人声，哼唱与歌曲人声不算口播。只输出 json。"""


def _classify_audio_by_script(analysis: dict) -> dict:
    """模型听不了音频时的兜底：用拆解结果里的台词与音乐字段推断。"""
    shots = analysis.get("分镜") or []
    speech = any(str(s.get("台词") or "").strip() not in ("", "无", "-") for s in shots)
    music_text = str((analysis.get("整体") or {}).get("音乐") or "")
    music = bool(music_text.strip()) and not music_text.strip().startswith(("无", "没有"))
    return {"有背景音乐": music, "有人声口播": speech, "音乐说明": music_text[:120] or "无",
            "人声说明": "分镜台词非空" if speech else "无", "置信度": 0.5,
            "判定来源": "拆解结果推断（音频模型不可用）"}


def _to_wav(src: str, dst: str, sr: int = 44100) -> str:
    """转成单声道 wav：librosa 装的 soundfile 不认 aac/m4a，必须先解码。"""
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                             "-ac", "1", "-ar", str(sr), dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("转 wav 失败：%s" % ret.stderr[-200:])
    return dst


# demucs（htdemucs）的人声/伴奏分离质量远好于 REPET-SIM（后者对参考片口播会留明显残响，
# 实测残留人声跟着 BGM 垫进成片能听出第二路口播）。权重已缓存在 ~/.cache/torch，离线可跑；
# 本机没 GPU，固定 -d cpu（37s 音频约 1~3 分钟，每任务只跑一两次，可接受）。
DEMUCS_PYTHON = os.getenv("DEMUCS_PYTHON", "/root/miniconda3/envs/liveclip_vocal/bin/python")
DEMUCS_TIMEOUT = int(os.getenv("DEMUCS_TIMEOUT", "900"))


def _demucs_separate(src: str, dst: str, want: str) -> dict:
    """htdemucs 两轨分离（vocals / no_vocals），want 决定取哪轨，编码到 dst。失败抛异常。"""
    if not os.path.isfile(DEMUCS_PYTHON):
        raise RuntimeError("demucs 解释器不存在：%s" % DEMUCS_PYTHON)
    base = os.path.splitext(dst)[0]
    wav_in = _to_wav(src, base + "_mix.wav")
    outdir = base + "_demucs"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    ret = subprocess.run([DEMUCS_PYTHON, "-m", "demucs.separate", "--two-stems", "vocals",
                          "-d", "cpu", "-n", "htdemucs", "-o", outdir, wav_in],
                         capture_output=True, text=True, timeout=DEMUCS_TIMEOUT, env=env)
    stem = os.path.join(outdir, "htdemucs",
                        os.path.splitext(os.path.basename(wav_in))[0],
                        "vocals.wav" if want == "vocal" else "no_vocals.wav")
    if ret.returncode != 0 or not os.path.isfile(stem):
        raise RuntimeError("demucs 分离失败：%s" % (ret.stderr or ret.stdout)[-200:])
    enc = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                              "-i", stem, "-c:a", "aac", "-b:a", "160k", dst])
    if enc.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("demucs 产物编码失败：%s" % enc.stderr[-200:])
    os.remove(wav_in)
    shutil.rmtree(outdir, ignore_errors=True)
    return {"file": dst, "method": "demucs_htdemucs_" + want}


def separate_bgm(src: str, dst: str, want: str = "bgm") -> dict:
    """从混合音轨里抽伴奏或人声，返回 {"file","method"}。want="bgm" 取伴奏，"vocal" 取人声。

    librosa 的 REPET-SIM 软掩蔽：人声是「非重复前景」，用余弦相似度中值滤波估出重复的
    背景（伴奏），再软掩蔽取背景（want="vocal" 时取前景）。纯 numpy 实现，不需要下载模型。
    失败时回落 ffmpeg：取伴奏用立体声中置抵消（左右声道相减，抵掉居中的人声），
    取人声只能退化成中置求和（抵不掉伴奏，只保证不丢人声）。
    """
    try:
        return _demucs_separate(src, dst, want)
    except Exception as exc:  # noqa: BLE001
        demucs_reason = str(exc)[:150]
    # numba 默认想把编译缓存写进 site-packages，只读环境会直接报 no locator available
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(config.OUTPUT_DIR, ".numba_cache"))
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        wav_in = _to_wav(src, os.path.splitext(dst)[0] + "_mix.wav")
        y, sr = librosa.load(wav_in, sr=None, mono=True)
        spec, phase = librosa.magphase(librosa.stft(y))
        # 切片通常只有几秒，2s 的相似度窗口会超过总帧数，nn_filter 直接报错；
        # 它要求 width < (帧数-1)//2，所以按帧数收窄（收不到 1 帧就走下面的 ffmpeg 回落）
        width = max(1, min(int(librosa.time_to_frames(2.0, sr=sr)),
                           (spec.shape[-1] - 1) // 2 - 1))
        bg = np.minimum(spec, librosa.decompose.nn_filter(spec, aggregate=np.median,
                                                          metric="cosine", width=width))
        fg = spec - bg
        if want == "vocal":
            mask, base = librosa.util.softmask(fg, 2.0 * bg, power=2), fg
        else:
            mask, base = librosa.util.softmask(bg, 2.0 * fg, power=2), bg
        wav_out = os.path.splitext(dst)[0] + ".wav"
        sf.write(wav_out, librosa.istft(mask * base * phase), sr)
        ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", wav_out, "-c:a", "aac", "-b:a", "160k", dst])
        os.remove(wav_in)
        if ret.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError("音轨编码失败：%s" % ret.stderr[-200:])
        return {"file": dst, "method": "librosa_repet_sim_" + want,
                "残留风险": True, "demucs不可用": demucs_reason}
    except Exception as exc:  # noqa: BLE001
        pan = ("pan=mono|c0=0.5*c0+0.5*c1" if want == "vocal"
               else "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c1-0.5*c0")
        ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                                 "-af", pan, "-c:a", "aac", "-b:a", "160k", dst])
        if ret.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError("音轨分离失败：%s / %s" % (str(exc)[:150], ret.stderr[-150:]))
        return {"file": dst, "method": "ffmpeg_pan_" + want, "残留风险": True,
                "demucs不可用": demucs_reason, "fallback_reason": str(exc)[:150]}


def _bgm_from_user(rec: dict, full: str, info: dict) -> str:
    return rec["inputs"].get("bgm") or ""


def _bgm_whole_track(rec: dict, full: str, info: dict) -> str:
    return info.get("候选BGM文件") or full


def _bgm_separated(rec: dict, full: str, info: dict) -> str:
    if info.get("候选BGM文件"):
        return info["候选BGM文件"]
    out = separate_bgm(full, _p(rec["task_id"], "audio", "bgm.m4a"))
    info["分离方法"] = out["method"]
    if out.get("残留风险"):
        # 弱分离器（REPET/声道抵消）分不干净人声：参考片有口播时残留会垫进成片，必须显式留痕
        info["伴奏残留人声风险"] = True
    if out.get("fallback_reason"):
        info["分离回落原因"] = out["fallback_reason"]
    return out["file"]


# 动作名（rules.REF_AUDIO_RULES 的「动作」）→ 怎么拿到贴回成片的 BGM 文件；None = 不贴
REF_AUDIO_ACTIONS = {"用户上传音乐": _bgm_from_user, "整轨复用": _bgm_whole_track,
                     "分离伴奏": _bgm_separated, "不使用参考片音频": None}


def _reference_audio_facts(rec: dict, full: str) -> dict:
    """参考片音轨的判定维度。拆解结果里已经直接给出就不再调音频模型。"""
    with open(_p(rec["task_id"], "reference", "analysis.json"), encoding="utf-8") as fh:
        analysis = json.load(fh)
    overall = analysis.get("整体") or {}
    info = {"有背景音乐": rules.as_bool(overall.get("有背景音乐")),
            "有人声口播": rules.as_bool(overall.get("有人声口播")),
            "音乐说明": str(overall.get("音乐") or "")[:120] or "无",
            "人声说明": str(overall.get("声音设计") or "")[:120] or "无",
            "判定来源": "拆解结果直接给出"}
    if info["有背景音乐"] is not None and info["有人声口播"] is not None:
        return info
    try:
        raw = aigc.vision_gemini(AUDIO_CLASS_PROMPT, media=[{"type": "audio", "url": full}])
        got = write_script._parse_json(raw)
        got.update({"有背景音乐": rules.as_bool(got.get("有背景音乐")),
                    "有人声口播": rules.as_bool(got.get("有人声口播")),
                    "判定来源": "拆解结果缺字段，补问音频模型"})
        return got
    except Exception as exc:  # noqa: BLE001
        log(rec, "音频模型不可用，改用拆解结果推断：%s" % str(exc)[:120])
        return _classify_audio_by_script(analysis)


def step_audio(rec: dict) -> dict:
    """按「参考片音轨」规则表判定，并准备好要贴回成片的 BGM 文件。"""
    tid = rec["task_id"]
    full = extract_audio(rec["inputs"]["reference_video"], _p(tid, "audio", "reference.m4a"))
    info = _reference_audio_facts(rec, full)
    user_bgm = rec["inputs"].get("bgm")

    # 复刻口径：参考片背景音只是零星音效/环境声（不是音乐）就不复用。先把将要贴的
    # 那条轨备出来（有口播先分离伴奏），对它判音乐性，结果作为事实进表。
    if (not (user_bgm and os.path.isfile(user_bgm))
            and rules.as_bool(info.get("有背景音乐")) is True
            and rules.as_bool(info.get("有人声口播")) is not None):
        cand = (full if rules.as_bool(info.get("有人声口播")) is False
                else _bgm_separated(rec, full, info))
        info["候选BGM文件"] = cand
        info["音乐性"] = gates.check("BGM音乐性", cand)
        log(rec, "BGM 音乐性：%s（%s）"
            % (_yn(info["音乐性"].get("通过")),
               info["音乐性"].get("判据") or info["音乐性"].get("原因", "")))

    d = rules.decide("参考片音轨",
                     {"用户上传BGM": bool(user_bgm and os.path.isfile(user_bgm)),
                      "参考片有BGM": info.get("有背景音乐"),
                      "参考片有口播": info.get("有人声口播"),
                      "BGM是音乐": (info.get("音乐性") or {}).get("通过")})
    info.update({"策略": d["动作"], "原因": d["说明"],
                 "规则表": d["规则表"], "命中行": d["命中行"], "判定事实": d["事实"]})
    action = REF_AUDIO_ACTIONS[d["动作"]]
    info["bgm文件"] = action(rec, full, info) if action else ""
    info["参考片音轨"] = full
    if info.get("bgm文件") and isinstance(info.get("音乐性"), dict):
        info["音乐性"]["使用"] = True   # 门禁对账锚点：贴走的 BGM 必须过了音乐性门禁

    path = _p(tid, "audio", "reference_audio.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    log(rec, "参考片音频：BGM=%s 口播=%s → %s（%s 第%d行）"
        % (_yn(info.get("有背景音乐")), _yn(info.get("有人声口播")), info["策略"],
           d["规则表"], d["命中行"]))
    return {"artifact": _rel(tid, path), "策略": info["策略"],
            "有背景音乐": bool(info.get("有背景音乐")), "有人声口播": bool(info.get("有人声口播"))}
