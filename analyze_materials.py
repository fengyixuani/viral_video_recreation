"""用户自有视频素材分析：把素材拆成可召回、可定位的语义片段。

跑法：修改 main() 里的 VIDEO 后执行 ``python3 analyze_materials.py``。
产物：output/material_analysis/{视频名}.json + {视频名}.md

与 analyze_reference.py 的区别：
  - analyze_reference.py 研究一条完整视频的剧本和镜头骨架；
  - 本文件建立用户素材库，重点是片段的时间范围、可见内容、动作、
    情绪、主体和使用条件，便于未来按剧本需求召回素材。

当前项目尚未接入向量数据库，因此每个片段会额外生成稳定的「召回文本」，
作为后续 embedding / 向量检索的输入。
"""
import json
import os
import sys
from typing import Any, Optional

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]

OUT_DIR = os.path.join(config.OUTPUT_DIR, "material_analysis")

SEGMENT_FIELDS = {
    "片段序号": ("index", "segment_index"),
    "开始时间": ("start", "起始时间"),
    "结束时间": ("end", "终止时间"),
    "时长秒": ("duration_sec", "duration"),
    "片段类型": ("type", "segment_type"),
    "画面": ("visual", "frame"),
    "主体": ("subjects", "characters", "people"),
    "动作": ("action", "actions"),
    "场景": ("scene", "setting"),
    "景别运镜": ("shot_camera", "camera", "camera_move"),
    "情绪氛围": ("emotion", "mood"),
    "声音台词": ("audio", "dialogue", "sound"),
    "视觉标签": ("visual_tags", "tags"),
    "内容标签": ("content_tags", "semantic_tags"),
    "可用场景": ("use_cases", "usage"),
    "限制条件": ("constraints", "notes"),
    "叙事功能": ("function", "role"),
}

OVERALL_FIELDS = {
    "素材概要": ("summary",),
    "主体标签": ("subject_tags", "subjects"),
    "场景标签": ("scene_tags", "scenes"),
    "动作标签": ("action_tags", "actions"),
    "情绪标签": ("emotion_tags", "emotions"),
    "适合内容": ("recommended_uses", "use_cases"),
    "不适合内容": ("avoid_uses", "avoid"),
}

# 声音克隆样本：整条素材只挑一段最长的干净主角人声，在素材理解这一次调用里一并产出，
# 不额外发请求。voice_dub._voice_candidates 把它当第一优先候选。
# 实测（博士耳塞 17 条）证明这四条约束缺一不可：
#   1. 不定义「主角」并列排除项 → 机舱安全须知广播拿 6 分、拍摄现场工作人员闲聊拿 5 分，
#      真拿去克隆就克出了播音员或场工的嗓子；
#   2. 不设段长下限 → 1s 碎片的转写会漂移（「看一下耳塞啥样」→「水星」）；
#   3. 不写「不许猜」→ 低音量段（实测 mean -34.6dB）模型直接编内容；
#   4. 没有「排除说明」→ 记 0 分时无从判断是真没有还是漏听，违背「显式失败」。
VOICE_FIELDS = {
    "有主角人声": ("有人声", "has_voice"),
    "人声类型": ("voice_type",),
    "开始秒": ("start", "开始时间"),
    "结束秒": ("end", "结束时间"),
    "时长秒": ("duration_sec", "duration"),
    "清晰度评分": ("评分", "score", "clarity"),
    "评分依据": ("reason",),
    "干扰因素": ("interference", "noise"),
    "该段转写": ("转写", "transcript"),
    "排除说明": ("excluded", "exclusion"),
}

# 与 voice_dub.VOICE_BASE_MIN 同一口径（seedance 参考音单段下限 2s，短于它整条请求被拒）。
# 不从 voice_dub import：media.py 已经 import 本模块，voice_dub → media 会成环。
# 改这里要同步改 voice_dub.VOICE_BASE_MIN。
VOICE_SEG_MIN_SEC = 2.0

# 判定维度：rules.py 的规则表直接读这些布尔字段，所以必须在这里一次性问清楚，
# 不能让下游再去「画面」「主体」这些自由文本里猜关键词（猜错就整段走错分支）。
# 判不出来的写 null，收敛成 None：规则表只会被通配吃掉，不会等值命中某条规则。
JUDGE_FIELDS = {
    "真人出镜": ("有真人", "real_person"),
    "真人出镜口播": ("有口播", "口播", "真人口播"),
    "有BGM": ("有背景音乐", "背景音乐", "bgm"),
    "有人声": ("人声", "有说话声"),
    "模特人脸出镜": ("人脸出镜", "露脸", "有人脸"),
}


SYSTEM = (
    "你是短视频素材库标注师。只依据实际看到和听到的内容回答，"
    "不确定的信息写「不确定」，不要臆测人物身份、品牌或事件。输出合法 JSON。"
)

PROMPT = """请把这条用户自有视频分析成可被未来剧本召回的素材片段，输出 JSON。

切分规则：
- 按内容完整性和可复用性切分，而不是机械地每秒切一段；
- 每个片段必须是连续时间区间，片段之间首尾相接、覆盖全片；
- 一个片段尽量表达一个完整动作或一个明确画面意图；
- 不要把同一连续动作切得过碎；没有明显切点时保留为一个片段；
- 片段类型从「人物表演、人物状态、产品展示、产品使用、环境空镜、
  细节特写、动作过程、结果展示、转场素材、其他」中选择。

每个「片段」必须包含以下字段：
- 片段序号、开始时间、结束时间、时长秒
- 片段类型
- 画面：构图、主体位置、光线、色调和画面风格
- 主体：数组，说明人物/物品/动物等主体及可识别外观
- 动作：动作起点、过程和结果；没有动作写「无」
- 场景：地点、环境、道具和时间/天气（看不出就写「不确定」）
- 景别运镜：景别、机位、运镜及方向速度
- 情绪氛围：主体情绪和整体氛围
- 声音台词：逐字记录能听清的人声，并写明音效、音乐、环境音；无则写「无」
- 视觉标签：适合按画面检索的短标签数组，例如「手部特写」「暖色」「慢动作」
- 内容标签：适合按语义检索的短标签数组，例如「打开包装」「展示质地」
- 可用场景：未来剧本中可以怎么使用，例如「开场钩子」「产品卖点展示」
- 限制条件：画面中不可替换或需要注意的内容，例如人物露脸、品牌字样；无则写「无」
- 叙事功能：钩子/铺垫/冲突/过程/高潮/结果/收尾/氛围等

以下 5 个字段必须是布尔值 true / false，实在判不出来才写 null（不要写字符串）：
- 真人出镜：画面里有没有真实人物出现
- 真人出镜口播：画面里的人有没有张嘴在说话（看得到嘴部开合，声音就是他发出的）；
  人物出镜但没张嘴（画面配旁白/画外音）、只有旁白配音画面里看不到说话的人 → false
- 有BGM：有没有持续的旋律或节奏配乐；环境音、音效不算
- 有人声：有没有能听出说话内容的人声；哼唱与歌曲人声不算
- 模特人脸出镜：有没有清晰可见、可用来做人物参考的正面或侧面人脸

「整体」必须包含：
- 素材概要、主体标签、场景标签、动作标签、情绪标签
- 适合内容、不适合内容

「主角人声」是整条素材只给一段的音色样本，用来做声音克隆，必须包含下列字段。
主角 = 这条素材里讲解商品、或者对着镜头说话的那个人（出镜口播和画外口播都算）。
以下声音一律**不算**主角人声，哪怕录得很清晰也必须排除：
- 机舱、车站、商场的公共广播与播报，安全须知，提示音
- 电视、手机、音响里播放出来的人声
- 路人、旁人、拍摄现场工作人员的闲聊或指挥
- 既听不出在讲商品、也听不出是对着镜头说的背景对话
在整条音轨里找**最长的一段清晰主角人声**：先保证清晰，再在清晰的前提下取最长。
不足 %.1f 秒的不算；整条都没有就把「有主角人声」记 false，起止与评分都记 0。
听不清就给低分，**绝对不要猜测或补全没听清的字**，宁可转写留空。
- 有主角人声：布尔值 true / false
- 人声类型：出镜口播 / 画外口播 / 无
- 开始秒、结束秒、时长秒：数字，必须落在 0 到全片时长之间
- 清晰度评分：0-10 整数。10 播音级近场干净单人声，无背景音乐与噪声，可直接做克隆样本；
  8-9 仅有轻微底噪，不影响任何字词；6-7 能完整听懂，有可察觉背景音但不掩盖人声；
  4-5 多数字能听懂但要专注，背景音明显；2-3 只听出零星词句；1 几乎听不到；0 没有主角人声
- 评分依据：30 字内
- 干扰因素：数组，从「背景音乐、环境噪声、混响、远场收音、爆音齿音、多人重叠、无」中选
- 该段转写：这一段主角说的原话，没听清的字宁可省略，没有就写「无」
- 排除说明：听到但被排除的声音及排除原因；没有就写「无」

严格只输出 JSON，顶层格式必须是：
{"片段": [...], "整体": {...}, "主角人声": {...}}
不要输出 markdown，不要省略字段。""" % VOICE_SEG_MIN_SEC


def _parse_json(raw: str) -> "dict[str, Any]":
    txt = raw.strip().strip("`")
    if txt.startswith("json"):
        txt = txt[4:]
    start, end = txt.find("{"), txt.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型没有返回 JSON 对象")
    return json.loads(txt[start:end + 1])


def _pick(src: "dict[str, Any]", cn: str, aliases: "tuple[str, ...]") -> Any:
    for key in (cn,) + aliases:
        value = src.get(key)
        if value or value == 0:
            return value
    return ""


def _sec(value) -> float:
    text = str(value).strip()
    if not text:
        return -1.0
    try:
        if ":" in text:
            result = 0.0
            for part in text.split(":"):
                result = result * 60 + float(part)
            return result
        return float(text)
    except (TypeError, ValueError):
        return -1.0


def _mmss(seconds: float) -> str:
    return "%02d:%04.1f" % (int(seconds // 60), seconds % 60)


def _video_duration(path: str) -> float:
    import subprocess
    import imageio_ffmpeg

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path],
        capture_output=True, text=True, check=False,
    ).stderr
    for line in result.splitlines():
        if "Duration:" not in line:
            continue
        try:
            hms = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
            hour, minute, second = hms.split(":")
            return int(hour) * 3600 + int(minute) * 60 + float(second)
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _as_bool(value) -> "Optional[bool]":
    """把模型返回的布尔/是否/有无收敛成布尔，判不出来返回 None（判定语义见 rules.as_bool）。"""
    return rules.as_bool(value)


def normalize(record: "dict[str, Any]") -> "dict[str, Any]":
    """将模型可能返回的中英文键名收敛到素材库 schema。"""
    raw_segments = record.get("片段") or record.get("segments") or []
    raw_overall = record.get("整体") or record.get("overall") or {}
    raw_voice = record.get("主角人声") or record.get("voice_sample") or {}
    segments: "list[dict[str, Any]]" = []
    for raw in raw_segments:
        if isinstance(raw, dict):
            segment = {
                cn: _pick(raw, cn, aliases)
                for cn, aliases in SEGMENT_FIELDS.items()
            }
            segment.update({
                cn: _as_bool(_pick(raw, cn, aliases))
                for cn, aliases in JUDGE_FIELDS.items()
            })
            segments.append(segment)
    overall = {
        cn: _pick(raw_overall, cn, aliases)
        for cn, aliases in OVERALL_FIELDS.items()
    }
    voice = {cn: _pick(raw_voice, cn, aliases) for cn, aliases in VOICE_FIELDS.items()}
    voice["有主角人声"] = _as_bool(voice.get("有主角人声"))
    return {"片段": segments, "整体": overall, "主角人声": voice}


def _fix_voice_sample(voice: "dict[str, Any]", real_total: float) -> "dict[str, Any]":
    """校验主角人声段的时间戳，落一个「可用」结论给下游读。

    模型自报的起止不能直接信：实测里它偶尔给出超过全片长度的结束秒，或者嘴上说有人声、
    起止却是 0-0。这里只做能本地判定的三件事（有没有、区间合不合法、够不够长），
    响度不在这里量——克隆基准音要量的是「将要使用的那个文件」，
    由 voice_dub 抽完 wav 后走 gates.check("克隆基准音")，这是 gates.py 第 1 条范式。
    不可用时写明原因，不静默清零。
    """
    start, end = _sec(voice.get("开始秒")), _sec(voice.get("结束秒"))
    span = round(end - start, 1) if start >= 0 and end > start else -1.0
    voice["开始秒"] = round(start, 1) if start >= 0 else 0.0
    voice["结束秒"] = round(end, 1) if end > 0 else 0.0
    voice["时长秒"] = span if span > 0 else 0.0
    if voice.get("有主角人声") is not True:
        voice.update({"可用": False, "弃用原因": "模型判定没有主角人声"})
        return voice
    if span <= 0:
        voice.update({"可用": False, "弃用原因": "起止时间非法（%s→%s）"
                                                % (voice["开始秒"], voice["结束秒"])})
        return voice
    if span < VOICE_SEG_MIN_SEC:
        voice.update({"可用": False, "弃用原因": "只有 %.1fs，短于样本下限 %.1fs"
                                                % (span, VOICE_SEG_MIN_SEC)})
        return voice
    if real_total > 0 and end > real_total * 1.05:
        voice.update({"可用": False, "弃用原因": "结束秒 %.1f 超出全片时长 %.1f"
                                                % (end, real_total)})
        return voice
    if real_total > 0:
        voice["结束秒"] = round(min(end, real_total), 1)
        voice["时长秒"] = round(voice["结束秒"] - voice["开始秒"], 1)
    voice.update({"可用": True, "弃用原因": ""})
    return voice


def _model_spans(segments: "list[dict[str, Any]]", real_total: float) -> "list[tuple]":
    """模型给的绝对起止时间，可信时返回 [(起, 止)]，不可信返回 []。

    可信 = 每段都有合法的 起 < 止、彼此不重叠且时间递增、末尾不超出真实时长。
    """
    spans = []
    for segment in segments:
        start, end = _sec(segment.get("开始时间")), _sec(segment.get("结束时间"))
        if not (start >= 0 and end > start):
            return []
        if spans and start < spans[-1][1] - 0.05:      # 与上一段重叠 → 不可信
            return []
        spans.append((start, end))
    if not spans:
        return []
    if real_total > 0 and spans[-1][1] > real_total * 1.05:
        return []
    return [(s, min(e, real_total) if real_total > 0 else e) for s, e in spans]


def _fix_timeline(segments: "list[dict[str, Any]]", real_total: float) -> "list[dict[str, Any]]":
    """优先使用模型给的绝对起止时间；只有它不可信时才按时长累加重建时间轴。

    以前是无条件用累加游标覆盖「开始时间」：模型没有严格首尾相接时（漏切片尾、跳过开头、
    片段之间有空隙——都很常见），所有片段的时间戳会整体前移，而且下面那段「时长总和与
    真实时长差 5% 就等比缩放」还会把每段一起拉长去凑总时长，把误差进一步放大。
    这份时间轴是下游一切「按秒定位」动作的唯一依据（product_images 的抽帧点校验、
    shot_match 的 开始秒、segment_build 的 cut_clip 窗口），偏移就等于抽错帧、裁错画面。
    """
    spans = _model_spans(segments, real_total)
    if spans:
        for index, (segment, (start, end)) in enumerate(zip(segments, spans), 1):
            segment["片段序号"] = index
            segment["开始时间"] = _mmss(start)
            segment["结束时间"] = _mmss(end)
            segment["时长秒"] = round(max(0.1, end - start), 1)
            segment["时间轴来源"] = "模型给定"
        return segments
    durations = []
    for segment in segments:
        start, end = _sec(segment.get("开始时间")), _sec(segment.get("结束时间"))
        duration = end - start if start >= 0 and end > start else _sec(segment.get("时长秒"))
        durations.append(max(0.1, duration) if duration >= 0 else 0.1)
    total = sum(durations)
    if real_total > 0 and total > 0 and abs(total - real_total) / real_total > 0.05:
        factor = real_total / total
        durations = [max(0.1, round(duration * factor, 1)) for duration in durations]
    cursor = 0.0
    for index, (segment, duration) in enumerate(zip(segments, durations), 1):
        segment["片段序号"] = index
        segment["时长秒"] = round(duration, 1)
        segment["开始时间"] = _mmss(cursor)
        cursor += duration
        segment["结束时间"] = _mmss(cursor)
        segment["时间轴来源"] = "按时长重排（模型起止时间不可用）"
    return segments


def _retrieval_text(segment: "dict[str, Any]") -> str:
    """生成后续 embedding 使用的自然语言描述，避免检索依赖某个字段。"""
    fields = ("片段类型", "画面", "主体", "动作", "场景", "情绪氛围",
              "视觉标签", "内容标签", "可用场景", "叙事功能")
    parts: "list[str]" = []
    for field in fields:
        value = segment.get(field)
        if value not in ("", None, [], "无"):
            parts.append("%s：%s" % (field, _format_value(value)))
    return "；".join(parts)


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return "，".join("%s%s" % (key, _format_value(item))
                         for key, item in value.items())
    return str(value)


def analyze(video: str, url: Optional[str] = None, engine: str = "gemini") -> "dict[str, Any]":
    if engine not in ("gemini", "qwen"):
        raise ValueError("engine 只支持 gemini 或 qwen")
    used, raw = engine, ""
    if engine == "gemini":
        try:
            raw = aigc.vision_gemini(PROMPT, media=[{"type": "video", "url": video}],
                                     system=SYSTEM)
        except (RuntimeError, OSError) as exc:
            print("gemini 不可用，回落 qwen：%s" % exc)
            used = "qwen"
    if used == "qwen":
        url = url or storage.upload(video)
        raw = aigc.vision(PROMPT, media=[{"type": "video", "url": url}],
                          system=SYSTEM, max_tokens=16384, json_mode=True)
    record = normalize(_parse_json(raw))
    real_total = _video_duration(video)
    record["片段"] = _fix_timeline(record["片段"], real_total)
    record["主角人声"] = _fix_voice_sample(record["主角人声"], real_total)
    stem = os.path.splitext(os.path.basename(video))[0]
    for segment in record["片段"]:
        segment["片段ID"] = "%s#%04d" % (stem, segment["片段序号"])
        segment["源视频"] = os.path.basename(video)
        segment["召回文本"] = _retrieval_text(segment)
    record["主角人声"]["源文件"] = video
    record["主角人声"]["素材ID"] = stem
    record.update({
        "素材ID": stem,
        "视频": os.path.basename(video),
        "来源URL": url or "",
        "链路": used,
        "总时长秒": round(real_total, 1),
        "召回版本": 1,
    })
    return record


def _fmt(value: Any) -> str:
    return _format_value(value) or "-"


def to_markdown(record: "dict[str, Any]") -> str:
    lines = ["# 素材分析：%s" % record.get("视频", ""), "", "## 整体", ""]
    for key, value in (record.get("整体") or {}).items():
        lines.append("- **%s**：%s" % (key, _fmt(value)))
    voice = record.get("主角人声") or {}
    lines += ["", "## 主角人声（声音克隆样本）", ""]
    if voice.get("可用"):
        lines.append("- **样本区间**：%.1f → %.1fs（%.1fs）" %
                     (voice.get("开始秒") or 0, voice.get("结束秒") or 0,
                      voice.get("时长秒") or 0))
        for key in ("人声类型", "清晰度评分", "评分依据", "干扰因素", "该段转写", "排除说明"):
            lines.append("- **%s**：%s" % (key, _fmt(voice.get(key))))
    else:
        lines.append("- **不可用**：%s" % _fmt(voice.get("弃用原因")))
        lines.append("- **排除说明**：%s" % _fmt(voice.get("排除说明")))
    lines += ["", "## 可召回片段", ""]
    for segment in record.get("片段") or []:
        lines.append("### 片段 %s  %s→%s（%ss）" %
                     (segment.get("片段序号"), segment.get("开始时间"),
                      segment.get("结束时间"), segment.get("时长秒")))
        for key in SEGMENT_FIELDS:
            if key != "片段序号":
                lines.append("- **%s**：%s" % (key, _fmt(segment.get(key))))
        lines.append("- **判定维度**：%s"
                     % "；".join("%s=%s" % (key, {True: "是", False: "否"}.get(segment.get(key),
                                                                              "不确定"))
                                 for key in JUDGE_FIELDS))
        lines.append("- **召回文本**：%s" % _fmt(segment.get("召回文本")))
        lines.append("")
    return "\n".join(lines)


def main():
    # ==== 调试参数：直接改这里 ====
    VIDEO = "/root/jmzhang/baidu/ViralForge/videos/others/13_泥膜棒_参考_理然泥膜棒.mp4"
    ENGINE = "gemini"  # gemini；不可用时自动回落 qwen
    # ==============================
    video = VIDEO if os.path.isabs(VIDEO) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), VIDEO)
    if not os.path.isfile(video):
        print("视频不存在:", video)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    print("分析用户素材中（engine=%s）..." % ENGINE)
    record = analyze(video, engine=ENGINE)
    stem = os.path.splitext(os.path.basename(video))[0]
    json_path = os.path.join(OUT_DIR, stem + ".json")
    md_path = os.path.join(OUT_DIR, stem + ".md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(record))
    print("片段数: %d ｜ 总时长 %.1fs" % (len(record["片段"]), record["总时长秒"]))
    print("产物:\n  %s\n  %s" % (json_path, md_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
