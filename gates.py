"""实测门禁：媒体资产用进成片前的可测事实检查（范式层，与 rules.py 表驱动决策同级）。

这一类问题的统一打法（不再在管线里散写 if + 魔法数）：

1. 标注是假设，不是事实。VLM/LLM 说「有人声」「有BGM」只配当候选依据；
   资产被用进成片前，必须对「将要使用的那个文件」实测（4496：标注有人声，
   实测 -42.6 dB 底噪；54f7：标注有BGM，实测是零星音效不是音乐）。
2. 判定进声明表。每个门禁 = 指标函数 + 阈值 + 判不出策略 + 口径说明，全部集中在
   GATES；调用方只喊 check(门禁名, 文件)，不自带阈值，改口径只动这一处。
3. 报告是标准件。{"门禁","通过","指标","判据","原因"}，原样写进产物 JSON 留痕；
   谁真用了这份资产就在报告上加 "使用": True。
4. 判不出 ≠ 不通过。探测手段挂了走门禁声明的「判不出」策略：
   「放行」= 宁可维持旧行为也不静默丢内容（如 BGM 音乐性判定器坏了不该丢 BGM）；
   「拦下」= 来路不明的资产不许上（如量不出响度的音频不许当克隆基准）。
   调用方统一用 ok(report) 拿最终放行结论，不自己解释 None。
5. 复核对账。review_case 扫全部产物里的门禁报告，「通过=False 且 使用=True」一律记
   问题——新加的门禁自动被对账覆盖，不用改复核器。

加一个新门禁 = GATES 里加一条声明（必要时补一个指标函数）+ 调用方一行 check()。
python3 gates.py 生成 output/gates.md 门禁清单（同 rules.py 的通览自检）。
"""
import os

import produce_video  # pyright: ignore[reportImplicitRelativeImport]

# ---------------- 阈值（全项目单一来源） ----------------
SILENT_DB = -45.0        # 静音线：平均音量低于它就当没有声音（seedance 偶尔返回静音轨）
VOICE_MIN_DB = -38.0     # 克隆基准音最低平均响度：比它弱的「人声」多半是底噪，克出来是噪音嗓
MUSIC_MIN_COVER = 0.5    # BGM 有声占比下限：音乐基本铺满，零星音效大段留白
MUSIC_MIN_RUN = 5.0      # BGM 最长连续有声段下限（秒）：音乐成段，音效是短促脉冲
MUSIC_MAX_FLATNESS = 0.35  # 谱平坦度中位数上限：音乐有音高/和声结构，环境底噪接近白噪


def _ffmpeg() -> str:
    return produce_video._ffmpeg()


# ---------------- 指标原语（纯 ffmpeg/numpy，可在无 GPU 机器上跑） ----------------
def mean_db(path: str) -> float:
    """平均音量（dB）；没有音轨或量不出来返回 -99（按静音处理）。"""
    err = produce_video._run([_ffmpeg(), "-hide_banner", "-i", path, "-vn",
                             "-af", "volumedetect", "-f", "null", "-"]).stderr
    for line in err.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0])
            except ValueError:
                break
    return -99.0


def silence_profile(path: str) -> dict:
    """有声占比与最长连续有声段（ffmpeg silencedetect，噪声门固定 -40dB / 0.4s）。

    噪声门必须是绝对阈值，不能随轨响度自适应——踩过的坑：零星音效轨（54f7）的间隙
    不是数字静音而是 -60dB 上下的环境底噪，门一放低整条轨都算「有声」，锚点 case 翻判。
    真正安静到 -40 以下的 BGM 贴回成片（原声直取不加增益）也听不见，拦下才是对的。"""
    dur = produce_video._duration(path)
    if dur <= 0.5:
        raise RuntimeError("音轨太短，判不出")
    noise = -40
    err = produce_video._run([_ffmpeg(), "-hide_banner", "-i", path, "-vn",
                             "-af", "silencedetect=noise=%ddB:d=0.4" % noise,
                             "-f", "null", "-"]).stderr
    silences, start = [], None
    for line in err.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip().split()[0])
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].strip().split()[0])
            silences.append((max(0.0, start), min(dur, end)))
            start = None
    if start is not None:
        silences.append((max(0.0, start), dur))
    cur, runs = 0.0, []
    for s, e in sorted(silences):
        if s > cur:
            runs.append((cur, s))
        cur = max(cur, e)
    if cur < dur:
        runs.append((cur, dur))
    active = sum(e - s for s, e in runs)
    return {"时长秒": round(dur, 2), "噪声门dB": noise,
            "有声占比": round(active / dur, 3),
            "最长连续有声秒": round(max((e - s for s, e in runs), default=0.0), 2)}


def spectral_flatness_median(path: str) -> float:
    """有声帧谱平坦度中位数（纯 numpy STFT；librosa 会拖起 numba 缓存，本机崩）。"""
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    tmp = path + ".flat.wav"
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-t", "90", "-i", path, "-ac", "1", "-ar", "22050",
                             "-c:a", "pcm_s16le", tmp])
    if ret.returncode != 0 or not os.path.isfile(tmp):
        raise RuntimeError("解码失败：%s" % ret.stderr[-100:])
    y, _sr = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    win, hop = 2048, 512
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < win:
        raise RuntimeError("有效采样太短，判不出")
    frames = np.lib.stride_tricks.sliding_window_view(y, win)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2
    rms = np.sqrt((frames ** 2).mean(axis=1))
    eps = 1e-10
    flat = np.exp(np.log(spec + eps).mean(axis=1)) / (spec.mean(axis=1) + eps)
    voiced = rms > max(rms.max() * 0.05, eps)
    return float(np.median(flat[voiced])) if voiced.any() else 1.0


# ---------------- 门禁声明表 ----------------
def _m_loudness(path: str) -> dict:
    return {"平均响度dB": round(mean_db(path), 1)}


def _m_music(path: str) -> dict:
    got = silence_profile(path)
    got["谱平坦度"] = round(spectral_flatness_median(path), 3)
    return got


GATES = {
    "有声内容": {
        "对象": "任何「应当有声」的音视频（AI 段音轨、分离出的人声等）",
        "指标": _m_loudness,
        "判定": lambda m: m["平均响度dB"] > SILENT_DB,
        "判据": lambda m: "平均响度 %.1f dB（静音线 %.0f）" % (m["平均响度dB"], SILENT_DB),
        "判不出": "拦下",
        "说明": "低于静音线就当没有声音；量不出响度的按无声处理",
    },
    "克隆基准音": {
        "对象": "TTS 克隆 / seedance 参考音用的基准人声文件",
        "指标": _m_loudness,
        "判定": lambda m: m["平均响度dB"] > VOICE_MIN_DB,
        "判据": lambda m: "平均响度 %.1f dB（门槛 %.0f）" % (m["平均响度dB"], VOICE_MIN_DB),
        "判不出": "拦下",
        "说明": "标注说有人声不代表真有（4496：-42.6dB 底噪标成人声）；弱参考克出噪音嗓，"
                "量不出的更不许当基准",
    },
    "BGM音乐性": {
        "对象": "将要贴进成片的参考片 BGM 轨（整轨或分离伴奏）",
        "指标": _m_music,
        "判定": lambda m: (m["有声占比"] >= MUSIC_MIN_COVER
                           and m["最长连续有声秒"] >= min(MUSIC_MIN_RUN, m["时长秒"] * 0.6)
                           and m["谱平坦度"] <= MUSIC_MAX_FLATNESS),
        "判据": lambda m: ("占比%.2f≥%.2f、连续%.1fs≥%.1fs、平坦度%.2f≤%.2f"
                           % (m["有声占比"], MUSIC_MIN_COVER, m["最长连续有声秒"],
                              min(MUSIC_MIN_RUN, m["时长秒"] * 0.6), m["谱平坦度"],
                              MUSIC_MAX_FLATNESS)),
        "判不出": "放行",
        "说明": "背景音只是零星音效/环境声（不是音乐）就不复用（54f7 宫斗茶：占比0.43）；"
                "判定器挂了按通配放行，不因它丢 BGM",
    },
}


# ---------------- 标准检查入口 ----------------
def check(name: str, path: str) -> dict:
    """跑一个门禁，返回标准报告：{"门禁","通过","指标","判据","原因","判不出"}。

    通过 = True/False/None（None = 探测手段挂了，按声明的「判不出」策略走，
    调用方用 ok() 拿最终放行结论）。报告原样写进产物；用了就补 "使用": True。
    """
    spec = GATES[name]
    out = {"门禁": name, "通过": None, "判不出": spec["判不出"]}
    if not (path and os.path.isfile(path)):
        return dict(out, 原因="文件不存在：%s" % (path or "(空)"))
    try:
        metrics = spec["指标"](path)
        out.update({"指标": metrics, "通过": bool(spec["判定"](metrics)),
                    "判据": spec["判据"](metrics)})
    except Exception as exc:  # noqa: BLE001
        out["原因"] = "探测失败：%s" % str(exc)[:120]
    return out


def ok(report: dict) -> bool:
    """报告 → 最终放行结论：判不出时按门禁声明的策略（放行/拦下）折算。"""
    if report.get("通过") is None:
        return report.get("判不出") == "放行"
    return bool(report["通过"])


def violations(obj, where: str = "") -> list:
    """在任意产物 JSON 结构里找「通过=False 却 使用=True」的门禁报告（复核对账用）。"""
    bad = []
    if isinstance(obj, dict):
        if obj.get("门禁") and obj.get("通过") is False and obj.get("使用") is True:
            bad.append("%s%s（%s）" % (where and where + "：", obj["门禁"],
                                       obj.get("判据") or obj.get("原因", "")))
        for v in obj.values():
            bad += violations(v, where)
    elif isinstance(obj, list):
        for v in obj:
            bad += violations(v, where)
    return bad


def overview() -> str:
    """门禁清单落盘 output/gates.md（文档即代码，同 rules.py 通览）。"""
    lines = ["# 实测门禁清单（gates.py 自动生成）", ""]
    for name, spec in GATES.items():
        lines += ["## %s" % name,
                  "- 对象：%s" % spec["对象"],
                  "- 判不出策略：%s" % spec["判不出"],
                  "- 说明：%s" % spec["说明"], ""]
    lines += ["## 阈值", "- SILENT_DB = %.1f" % SILENT_DB,
              "- VOICE_MIN_DB = %.1f" % VOICE_MIN_DB,
              "- MUSIC_MIN_COVER / MIN_RUN / MAX_FLATNESS = %.2f / %.1fs / %.2f"
              % (MUSIC_MIN_COVER, MUSIC_MIN_RUN, MUSIC_MAX_FLATNESS), ""]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "gates.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


if __name__ == "__main__":
    print("门禁 %d 个：%s" % (len(GATES), "、".join(GATES)))
    print("清单：%s" % overview())
