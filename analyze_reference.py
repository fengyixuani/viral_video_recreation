"""参考视频剧本拆解：用 VLM 把视频拆成分镜，并总结创意/结构/音乐情绪/节奏骨架。

跑法：python3 analyze_reference.py（待拆解视频改 main() 里的 VIDEO，链路改 ENGINE）
产物：output/script_analysis/{视频名}.json + {视频名}.md

字段名统一用中文（分镜/整体/台词/音效音乐…），模型看中文键名更不容易漏字段；
老产物里的英文键名由 normalize() 自动映射过来，下游读中文键即可。

默认 engine=gemini：走 aigc.vision_gemini()（内网 openairr 网关，视频 base64 内联，不用先上传）；
gemini 不通时自动回落 qwen：aigc.vision()，先 storage.upload() 成公网 URL 再让模型看视频。
"""
import json
import os
import sys
from typing import Any, Optional

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]

OUT_DIR = os.path.join(config.OUTPUT_DIR, "script_analysis")

# 分镜字段：中文键名 -> 兼容的旧英文键与常见别名（模型偶发改名，一律收敛回中文键）
SHOT_FIELDS = {
    "序号": ("index",),
    "开始时间": ("起", "start", "起始时间"),
    "结束时间": ("止", "end", "终止时间"),
    "时长秒": ("duration_sec", "duration"),
    "景别": ("shot_size", "size"),
    "运镜": ("camera", "camera_move"),
    "画面": ("visual", "frame"),
    "人物": ("characters", "people"),
    "动作": ("action", "actions"),
    "场景": ("scene", "setting"),
    "特效": ("vfx", "effects"),
    # 「转场」按 PROMPT 的定义是**从上一镜进入这一镜**的方式，所以只收「转入」这一侧。
    # 把 转出方式/transition_out 也当别名的话，模型只写转出时整份转场会错位一镜，
    # 而这个字段正是 write_script._is_hard_cut 决定在哪里断段的唯一依据。
    "转场": ("转入方式", "转入", "转场方式", "transition_in"),
    "台词": ("chat", "dialogue", "dialog", "lines", "speech"),
    "音效音乐": ("audio", "sound", "sfx", "sound_design"),
    "叙事功能": ("function", "role"),
}

# 整体字段：同上
OVERALL_FIELDS = {
    "一句话概要": ("logline",),
    "核心创意": ("creative_idea", "idea"),
    "开场钩子": ("hook",),
    "结构": ("structure",),
    "叙事技巧": ("narrative_devices", "devices"),
    "音乐": ("music",),
    "声音设计": ("sound_design",),
    "情绪曲线": ("emotion_curve",),
    "节奏": ("rhythm",),
    "镜头骨架": ("shot_skeleton", "skeleton"),
    "可复用模板": ("reusable_template", "template"),
    "产品植入": ("product_placement", "placement"),
}

# 判定维度：参考片音轨怎么复刻由 rules.py 的「参考片音轨」表决定，表要读布尔值，
# 所以拆解时就一次问清楚，省掉后面再单独调一次音频模型；判不出来写 null。
JUDGE_FIELDS = {
    "有背景音乐": ("有BGM", "背景音乐", "bgm"),
    "有人声口播": ("有口播", "口播", "有人声"),
}

SYSTEM = ("你是短剧/短视频的分镜拆解师和创意分析师。只依据你实际看到听到的画面与声音回答，"
          "不确定的字段写「不确定」而不要编造。输出 json。")

PROMPT = """请把这条视频当作一个可被复刻的剧本来拆解，输出 json。所有字段名用下面给出的中文原文。

一、逐个分镜拆（"分镜" 数组，按时间顺序，不要漏镜头，也不要把一个镜头拆成多条）：
每个分镜给出：
- 序号：从 1 开始
- 开始时间 / 结束时间：这一镜在原片里的起止时间，格式 "mm:ss.s"，必须逐镜首尾相接覆盖全片
- 时长秒：时长（秒，可带小数）
- 景别：大远景/远景/全景/中景/近景/特写/大特写
- 运镜：机位与运镜（固定/推/拉/摇/移/跟/升降/手持晃动/环绕/变焦，含运动方向与速度）
- 画面：这一镜的画面整体是什么样子，一段话讲清构图、光线、色调、画质风格
- 人物：数组，每人给 谁、外形服装、表情情绪、视线朝向；无人物写空数组
- 动作：人物或主体在做什么动作，动作的起点和终点
- 场景：场景与环境（地点、时代/风格、关键道具陈设、天气时间）
- 特效：滤镜、调色、慢动作/加速、抖动、粒子、光效、字幕花字、贴纸、遮罩等；无则空数组
- 转场：从上一镜进入这一镜的转场方式（硬切/叠化/闪白闪黑/划像/遮罩转场/运镜衔接/匹配剪辑/无），
  第 1 镜写「无」
- 台词：这一镜听到的人物台词、旁白、口播，逐字转写，写成「谁：原话」；确实没有人声才写「无」
- 音效音乐：这一镜除人声以外的声音（音效、音乐起落与变化、环境音）
- 叙事功能：这一镜在叙事里承担什么功能（钩子/铺垫/冲突升级/反转/情绪高点/落点/产品露出等）

二、整体分析（"整体" 对象）：
- 一句话概要：一句话讲清这条视频讲了什么
- 核心创意：核心创意是什么，为什么这个点子能抓人（矛盾点/反差/情绪杠杆在哪）
- 开场钩子：开头3s详细分析，如何制造钩子的，作用是什么，冲突和矛盾或者反差在哪
- 结构：整体结构，分段给出（每段：名称、时间区间、作用），例如 钩子—建立—冲突—反转—落点
- 叙事技巧：用到的叙事技巧（悬念、误导、身份反差、重复递进、call back 等）
- 音乐：风格/配器、情绪、入点与切点、鼓点与画面的对应关系、有无卡点
- 声音设计：音效与人声的处理特点
- 情绪曲线：按时间给出若干采样点（时间、情绪、强度 1-5）
- 节奏：总时长、镜头总数、平均镜头时长、最短/最长镜头、剪辑快慢的分布规律
- 镜头骨架：把整条片子抽象成可复用的镜头序列模板，每项给出 作用、景别、时长秒、备注
- 可复用模板：如果要换一个题材复刻这条片子，需要保留哪些骨架、可以替换哪些元素
- 产品植入：产品/卖点如何植入（无则写「无」）
- 有背景音乐：布尔值 true/false，全片有没有持续的旋律或节奏配乐（环境音、音效不算音乐）
- 有人声口播：布尔值 true/false，有没有能听出说话内容的人声台词或旁白（哼唱与歌曲人声不算）
上面两个布尔字段实在判不出来才写 null，不要写字符串。

每个分镜对象必须包含上面全部 15 个字段，一个都不能省（信息缺失也要给出判断或写「无」），
其中「转场」最容易被漏掉，请逐镜检查后再输出。
字段名必须原样使用上面的中文键名，不要改写、不要合并、不要漏字段；
「台词」与「音效音乐」是两个独立字段，不能互相替代，也不要把台词塞进整体分析里。
严格只输出 json，顶层为 {"分镜": [...], "整体": {...}}。"""


def _parse_json(raw: str) -> "dict[str, Any]":
    txt = raw.strip().strip("`")
    if txt.startswith("json"):
        txt = txt[4:]
    return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])


def _pick(src: "dict[str, Any]", cn: str, aliases: "tuple[str, ...]") -> Any:
    """按中文键取值，取不到再按别名/旧英文键取，都没有返回空串。"""
    for k in (cn,) + aliases:
        v = src.get(k)
        if v or v == 0:
            return v
    return ""


def _video_duration(path: str) -> float:
    """读视频真实时长（秒）。ffmpeg 取自 imageio_ffmpeg，无需系统安装；取不到返回 0。"""
    import subprocess

    import imageio_ffmpeg
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
                         capture_output=True, text=True, check=False).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            try:
                h, m, sec = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(sec)
            except ValueError:
                return 0.0
    return 0.0


def _sec(t: Any) -> float:
    """把 "mm:ss.s" / "0:03" / 秒数 解析成秒；解析不了返回 -1。"""
    txt = str(t).strip()
    if not txt:
        return -1.0
    try:
        if ":" in txt:
            parts = [float(x) for x in txt.split(":")]
            out = 0.0
            for v in parts:
                out = out * 60 + v
            return out
        return float(txt)
    except ValueError:
        return -1.0


def _fix_durations(shots: "list[dict[str, Any]]", real_total: float) -> "list[dict[str, Any]]":
    """校准每镜时长：先用起止时间反推，再按视频真实总时长等比缩放。

    模型自报的「时长秒」系统性偏小（实测 9 镜合计 6.6s，实际 12.2s），
    仿写剧本按这个时长复刻会让成片短一半，所以这里必须以真实时长为准。
    """
    for sh in shots:
        a, b = _sec(sh.get("开始时间")), _sec(sh.get("结束时间"))
        if a >= 0 and b > a:
            sh["时长秒"] = round(b - a, 1)
    total = sum(max(0.0, _sec(sh.get("时长秒"))) for sh in shots)
    if real_total > 0 and total > 0 and abs(total - real_total) / real_total > 0.05:
        k = real_total / total
        for sh in shots:
            sh["时长秒"] = round(max(0.1, _sec(sh.get("时长秒")) * k), 1)
    cur = 0.0                      # 时长变了，时间轴重算一遍保持自洽
    for sh in shots:
        dur = max(0.0, _sec(sh.get("时长秒")))
        sh["开始时间"], sh["结束时间"] = _mmss(cur), _mmss(cur + dur)
        cur += dur
    return shots


def _mmss(t: float) -> str:
    return "%02d:%04.1f" % (int(t // 60), t % 60)


def _fill_timeline(shots: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
    """模型漏写起止时间时，按各镜时长累加补齐，保证时间轴连续可读。"""
    cur = 0.0
    for s in shots:
        try:
            dur = float(s.get("时长秒") or 0)
        except (TypeError, ValueError):
            dur = 0.0
        if not str(s.get("开始时间")).strip():
            s["开始时间"] = _mmss(cur)
            s["结束时间"] = _mmss(cur + dur)
        cur += dur
    return shots


def normalize(rec: "dict[str, Any]") -> "dict[str, Any]":
    """把任意一版键名（中文 / 旧英文 / 模型改过的别名）收敛成中文 schema。"""
    shots_raw = rec.get("分镜") or rec.get("shots") or []
    overall_raw = rec.get("整体") or rec.get("overall") or {}
    shots: "list[dict[str, Any]]" = []
    for s in shots_raw:
        if isinstance(s, dict):
            shots.append({cn: _pick(s, cn, al) for cn, al in SHOT_FIELDS.items()})
    overall = {cn: _pick(overall_raw, cn, al) for cn, al in OVERALL_FIELDS.items()}
    overall.update({cn: rules.as_bool(_pick(overall_raw, cn, al))
                    for cn, al in JUDGE_FIELDS.items()})
    return {"分镜": _fill_timeline(shots), "整体": overall}


TRANSITION_PROMPT = """这条视频共有 __N__ 个镜头，按时间顺序分别是：
__LIST__

只判断每一镜是怎么从上一镜切进来的（第 1 镜写「无」），输出 json：
{"转场": [{"序号": 1, "转场": "..."}, ...]}
可选值：硬切/叠化/闪白/闪黑/划像/遮罩转场/运镜衔接/匹配剪辑/无。只输出 json。"""


def _fill_transitions(rec: "dict[str, Any]", video: str, engine: str) -> "dict[str, Any]":
    """转场字段是分段拼接的依据，长 schema 下常被漏写，缺了就单独补问一次。"""
    shots = rec["分镜"]
    missing = [s for s in shots if not str(s.get("转场")).strip()]
    if not shots or not missing or not engine == "gemini":
        return rec
    listing = "\n".join("%s. %s｜%s｜%s" % (s.get("序号"), s.get("开始时间"),
                                           s.get("景别"), str(s.get("画面"))[:40])
                         for s in shots)
    try:
        raw = aigc.vision_gemini(TRANSITION_PROMPT.replace("__N__", str(len(shots)))
                                 .replace("__LIST__", listing),
                                 media=[{"type": "video", "url": video}])
        got = {str(t.get("序号")): t for t in (_parse_json(raw).get("转场") or [])}
    except (RuntimeError, OSError, ValueError) as exc:
        print("转场补问失败，按硬切处理：%s" % str(exc)[:120])
        return rec
    for sh in shots:
        t = got.get(str(sh.get("序号"))) or {}
        sh["转场"] = sh.get("转场") or t.get("转场") or ""
    return rec


def _ask_json(ask, retries: int = 2) -> "dict[str, Any]":
    """调 ask(extra) 拿 json 并解析；格式坏了带着报错重问，最多 retries 次。

    拆解是整条链路里最贵的一次调用（一次视频理解），不能因为模型一次语法抖动
    （缺冒号、尾逗号、被 max_tokens 截断）就让整个步骤失败重跑——write_script
    早就为同类问题写了 _understand_json，这条路径漏了。
    """
    extra = ""
    for attempt in range(retries + 1):
        raw = ask(extra)
        try:
            return _parse_json(raw)
        except ValueError as exc:
            if attempt >= retries:
                raise
            print("拆解 json 解析失败（%s），重问一次" % str(exc)[:80], flush=True)
            extra = ("\n\n注意：上一次输出的 json 有语法错误（%s），"
                     "这次必须输出完整、合法、可被 json.loads 解析的 json。" % str(exc)[:120])


def analyze(video: str, url: Optional[str] = None, engine: str = "gemini") -> "dict[str, Any]":
    """拆解一条视频，返回 {"视频","来源URL","链路","分镜","整体"}。

    engine="gemini"：aigc.vision_gemini()，wenchain 内网 openairr 网关，
    视频 base64 内联，不必先 upload；engine="qwen"：aigc.vision()，需要公网 URL。
    gemini 不通时自动回落 qwen，实际用的链路记在「链路」字段里。
    """
    state = {"used": engine}

    def ask(extra: str) -> str:
        """问一次拆解（extra 是重问时追加的纠错提示）。gemini 不通就就地回落 qwen。

        回落判断放在这里而不是先探一次：拆解是一次完整的视频理解调用，
        先探再问等于把整条链路里最贵的那一步跑两遍。
        """
        nonlocal url
        if state["used"] == "gemini":
            try:
                return aigc.vision_gemini(PROMPT + extra,
                                          media=[{"type": "video", "url": video}], system=SYSTEM)
            except (RuntimeError, OSError) as exc:
                print("gemini 不可用，回落 qwen：%s" % exc)
                state["used"] = "qwen"
        url = url or storage.upload(video)
        return aigc.vision(PROMPT + extra, media=[{"type": "video", "url": url}],
                           system=SYSTEM, max_tokens=16384, json_mode=True)

    rec = _fill_transitions(normalize(_ask_json(ask)), video, state["used"])
    used = state["used"]
    real = _video_duration(video)
    rec["分镜"] = _fix_durations(rec["分镜"], real)
    rec.update({"视频": os.path.basename(video), "来源URL": url or "", "链路": used,
                "总时长秒": round(real, 1)})
    return rec


def _fmt(v: Any) -> str:
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return "\n" + "\n".join("  - " + "；".join("%s: %s" % (k, x[k]) for k in x) for x in v)
        return "、".join(str(x) for x in v) or "-"
    if isinstance(v, dict):
        return "\n" + "\n".join("  - %s: %s" % (k, _fmt(x)) for k, x in v.items())
    return str(v) or "-"


def to_markdown(rec: "dict[str, Any]") -> str:
    lines = ["# 剧本拆解：%s" % rec.get("视频", ""), "", "## 整体", ""]
    for k, v in (rec.get("整体") or {}).items():
        lines.append("- **%s**：%s" % (k, _fmt(v)))
    lines += ["", "## 分镜", ""]
    for s in rec.get("分镜") or []:
        lines.append("### 镜 %s  %s→%s（%ss）" % (s.get("序号"), s.get("开始时间"),
                                                s.get("结束时间"), s.get("时长秒")))
        for k in list(SHOT_FIELDS)[4:]:   # 序号/起/止/时长秒 已写在标题里
            lines.append("- **%s**：%s" % (k, _fmt(s.get(k))))
        lines.append("")
    return "\n".join(lines)


def main():
    # ==== 调试参数：直接改这里 ====
    # VIDEO = "/root/jmzhang/baidu/ViralForge/videos/drama/ref_videos/08_宫斗_参考_茉莉茶.mp4"   # 待拆解视频，相对本文件目录或绝对路径
    # VIDEO = '/root/jmzhang/baidu/ViralForge/videos/others/14_眼睛_参考_眼睛.mp4'
    # VIDEO = '/root/jmzhang/baidu/ViralForge/videos/others/04_项链_参考_xianglian.mp4'
    VIDEO = "/root/jmzhang/baidu/ViralForge/videos/others/13_泥膜棒_参考_理然泥膜棒.mp4"

    ENGINE = "gemini"   # gemini（默认，wenchain 内网 openairr）/ qwen（wenchain ali-qwen3.7-plus）
    # ==============================

    video = VIDEO if os.path.isabs(VIDEO) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), VIDEO)
    if not os.path.isfile(video):
        print("视频不存在:", video)
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    print("拆解中（engine=%s，视频较长会等一会）..." % ENGINE)
    rec = analyze(video, engine=ENGINE)

    stem = os.path.splitext(os.path.basename(video))[0]
    json_path = os.path.join(OUT_DIR, stem + ".json")
    md_path = os.path.join(OUT_DIR, stem + ".md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(rec))

    print("实际链路: %s" % rec["链路"])
    # 产物已经落盘了，别让一行统计把返回码变成异常退出：模型既没给时长秒、
    # ffmpeg 也读不到真实时长时（_fix_durations 两处赋值都不执行），时长秒会是空串。
    print("分镜数: %d ｜ 总时长 %.1fs（各镜合计 %.1fs）"
          % (len(rec["分镜"]), rec["总时长秒"],
             sum(max(0.0, _sec(s.get("时长秒"))) for s in rec["分镜"])))
    print("有台词的镜头: %d" % sum(1 for s in rec["分镜"]
                              if str(s.get("台词")).strip() not in ("", "无", "-")))
    print("产物:\n  %s\n  %s" % (json_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
