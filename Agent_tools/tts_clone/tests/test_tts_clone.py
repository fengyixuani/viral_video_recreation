#!/usr/bin/env python3
"""tts_clone 的自测：真实合成 + ASR 回听校验 + 错误路径，产出 samples/*.wav 与 TEST_REPORT.md。

用法（任意 python3 都行，模型/ASR 都在子进程里跑）:
    python tests/test_tts_clone.py
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import tts_clone  # noqa: E402

SAMPLES = os.path.join(_HERE, "samples")
REPORT = os.path.join(_HERE, "TEST_REPORT.md")
FFMPEG = tts_clone.FFMPEG
MATERIAL = "/home/wanghequan/Viral_Video_Agent/uploads"
# 两个真实参考音：一个录得正常(-23.8dB)、一个录得很轻(-39.2dB)，后者用来验响度归一。
REF_LOUD = os.path.join(MATERIAL, "be0a0f489f48ad82_干发慕斯-素材1.MP4")
REF_LOUD_TEXT = "头发一油就紧贴头皮看着太显脸大了"
REF_LOUD_RANGE = "0.0-8.0"
REF_QUIET = os.path.join(MATERIAL, "d3c75cf0ec9a3696_干发慕斯-素材4.MOV")
REF_QUIET_TEXT = "OK然后捏一捏那个泡沫"
REF_QUIET_RANGE = "5.39-7.96"
SHORT_TEXT = "洗完头喷一点，发根立马蓬起来"
LONG_TEXT = "头发一油就塌，抓两下发根，蓬松度立马回来，出门前三十秒搞定"
QUIET_TEXT = "按一下不塌，支撑力真的够"
# 回听校验用的本地 ASR（缺了就跳过，不影响其他断言）
ASR_PYTHON = os.getenv("ASR_PYTHON", "/root/miniconda3/envs/viral-split-asr/bin/python")
ASR_SCRIPT = os.getenv(
    "ASR_SCRIPT",
    "/home/wanghequan/Viral_Video_Split/common/vendor/Viral_Video/run_qwen3_asr_test.py")


def probe(path):
    """(时长, 采样率, 声道, mean_dB, max_dB)"""
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", path, "-af", "volumedetect",
                          "-f", "null", os.devnull],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).stderr.decode("utf-8", "ignore")
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "stream=sample_rate,channels", "-show_entries", "format=duration",
                           "-of", "default=nw=1", path],
                          stdout=subprocess.PIPE).stdout.decode("utf-8", "ignore")
    got = dict(line.split("=", 1) for line in info.splitlines() if "=" in line)

    def grab(key):
        for line in out.splitlines():
            if key in line:
                return line.split(key)[1].strip().split(" ")[0]
        return "?"
    return (round(float(got.get("duration", 0) or 0), 2), got.get("sample_rate", "?"),
            got.get("channels", "?"), grab("mean_volume:"), grab("max_volume:"))


def asr(path):
    """把产出念的内容转写回来（校验「念的就是给的文案」）。拿不到 ASR 返回 ""。"""
    if not (os.path.exists(ASR_PYTHON) and os.path.exists(ASR_SCRIPT)):
        return ""
    dst = path + ".asr.json"
    r = subprocess.run([ASR_PYTHON, ASR_SCRIPT, path, dst],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.isfile(dst):
        return ""
    try:
        with open(dst, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return ""
    finally:
        try:
            os.remove(dst)
        except OSError:
            pass

    def pick(o):
        if isinstance(o, dict):
            if isinstance(o.get("text"), str) and o["text"].strip():
                return o["text"].strip()
            for v in o.values():
                got = pick(v)
                if got:
                    return got
        if isinstance(o, list):
            for v in o:
                got = pick(v)
                if got:
                    return got
        return ""
    return pick(data)


def excerpt(src, rng, dst):
    """把参考音那一段抽出来存成 wav，方便和克隆产物 A/B 对听。"""
    a, b = tts_clone._parse_range(rng)
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", "{:.3f}".format(a), "-i", src,
                    "-t", "{:.3f}".format(max(0.5, b - a)),
                    "-vn", "-ac", "1", "-ar", "44100", dst],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_case(name, desc, want_text="", **kw):
    t0 = time.time()
    res = tts_clone.clone(**kw)
    row = {"name": name, "desc": desc, "cost_s": round(time.time() - t0, 1),
           "want_text": want_text, "res": res}
    if res.get("ok"):
        row["probe"] = probe(res["output"])
        row["asr"] = asr(res["output"])
    print("[{}] {} -> ok={} ({}s) {}".format(name, desc, res.get("ok"), row["cost_s"],
                                            row.get("asr", "")), flush=True)
    return row


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    excerpt(REF_LOUD, REF_LOUD_RANGE, os.path.join(SAMPLES, "ref_loud_原声.wav"))
    excerpt(REF_QUIET, REF_QUIET_RANGE, os.path.join(SAMPLES, "ref_quiet_原声.wav"))
    loud = dict(ref_audio=REF_LOUD, ref_text=REF_LOUD_TEXT, ref_range=REF_LOUD_RANGE)
    quiet = dict(ref_audio=REF_QUIET, ref_text=REF_QUIET_TEXT, ref_range=REF_QUIET_RANGE)
    rows = [
        run_case("C1", "默认(basic) + 短文案 14 字", SHORT_TEXT, text=SHORT_TEXT,
                 out_wav=os.path.join(SAMPLES, "C1_短文案.wav"), **loud),
        run_case("C2", "默认(basic) + 长文案 29 字", LONG_TEXT, text=LONG_TEXT,
                 out_wav=os.path.join(SAMPLES, "C2_长文案.wav"), **loud),
        run_case("C3", "极轻参考音(-39.2dB) + 默认归一 -16 LUFS", QUIET_TEXT, text=QUIET_TEXT,
                 out_wav=os.path.join(SAMPLES, "C3_轻参考_已归一.wav"), **quiet),
        run_case("C4", "同 C3 但 lufs=off（对照：不归一有多轻）", QUIET_TEXT, text=QUIET_TEXT,
                 lufs="off", out_wav=os.path.join(SAMPLES, "C4_轻参考_未归一.wav"), **quiet),
        run_case("C5", "prompt_mode=ultimate（对照：为什么默认不用它）", SHORT_TEXT,
                 text=SHORT_TEXT, prompt_mode="ultimate",
                 out_wav=os.path.join(SAMPLES, "C5_ultimate模式.wav"), **loud),
    ]
    never = os.path.join(SAMPLES, "never.wav")
    rows.append(run_case("E1", "参考文件不存在", ref_audio="/nope.mp4", ref_text="x",
                         text="y", out_wav=never))
    rows.append(run_case("E2", "ultimate 模式缺 ref_text", ref_audio=REF_LOUD, ref_text="",
                         text="y", prompt_mode="ultimate", out_wav=never))
    keep = tts_clone.TTS_SCRIPT
    tts_clone.TTS_SCRIPT = "/nope.py"
    rows.append(run_case("E3", "后端脚本路径错（未就绪）", ref_audio=REF_LOUD,
                         ref_text=REF_LOUD_TEXT, text="y", out_wav=never))
    tts_clone.TTS_SCRIPT = keep
    write_report(rows)
    synth_ok = all(r["res"].get("ok") for r in rows if r["name"].startswith("C"))
    err_ok = not any(r["res"].get("ok") for r in rows if r["name"].startswith("E"))
    print("\n合成用例全部产出：{} | 错误用例全部拦住：{}".format(synth_ok, err_ok))
    print("报告：{}".format(REPORT))
    return 0 if (synth_ok and err_ok) else 1


def write_report(rows):
    L = ["# tts_clone 自测报告", "",
         "跑法：`python tests/test_tts_clone.py`（本文件由脚本自动生成）", "",
         "- 后端：`{}`，参考音一律抽成 {}Hz 单声道".format(
             os.path.basename(tts_clone.TTS_SCRIPT), tts_clone.PROMPT_SR),
         "- 参考音 A（正常，整片 mean -23.8dB）：`{}` 取 {}s，原话「{}」".format(
             os.path.basename(REF_LOUD), REF_LOUD_RANGE, REF_LOUD_TEXT),
         "- 参考音 B（很轻，整片 mean -39.2dB）：`{}` 取 {}s，原话「{}」".format(
             os.path.basename(REF_QUIET), REF_QUIET_RANGE, REF_QUIET_TEXT),
         "- 「ASR 回听」= 用本地 Qwen3-ASR 把产出转写回来，校验**念的就是给的文案**"
         "（同音错字属 ASR 误识，不是合成错）", ""]
    L += ["## 合成结果", ""]
    for r in rows:
        res = r["res"]
        if not res.get("ok"):
            continue
        d, sr, ch, mean, mx = r["probe"]
        L += ["**{} {}**".format(r["name"], r["desc"]), "",
              "- 产物：`samples/{}` | 时长 {}s | {} Hz / {} 声道 | mean {} dB / max {} dB".format(
                  os.path.basename(res["output"]), d, sr, ch, mean, mx),
              "- 给的文案：「{}」".format(r["want_text"]),
              "- ASR 回听：「{}」".format(r.get("asr") or "（本机没跑 ASR）"),
              "- 响度归一：{} | 耗时 {}s（含模型加载）".format(
                  "是" if res.get("lufs_normalized") else "否（lufs=off）", r["cost_s"]), ""]
    L += ["## 错误路径（都应 ok=false + 中文原因，不抛异常）", ""]
    for r in rows:
        if r["res"].get("ok"):
            continue
        L += ["- **{} {}** → `{}`".format(r["name"], r["desc"], r["res"].get("error", "")[:120])]
    L += ["", "## 怎么听", "",
          "`tests/samples/` 里放了参考原声和克隆产物，按对听：", "",
          "- 音色像不像：`ref_loud_原声.wav` ↔ `C1_短文案.wav` / `C2_长文案.wav`",
          "- 轻参考音也能听清：`ref_quiet_原声.wav` ↔ `C3_轻参考_已归一.wav`",
          "- 为什么必须归一：`C3_轻参考_已归一.wav` ↔ `C4_轻参考_未归一.wav`"
          "（同文案同参考，只差 `lufs=off`）",
          "- 为什么默认 basic：`C1_短文案.wav` ↔ `C5_ultimate模式.wav`"
          "（ultimate 会把参考音原话也念出来）", "",
          "原始结果 JSON：", "", "```json",
          json.dumps([{"name": r["name"], "cost_s": r["cost_s"], "asr": r.get("asr", ""),
                       **r["res"]} for r in rows], ensure_ascii=False, indent=2), "```", ""]
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
