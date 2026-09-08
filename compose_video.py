"""拼接与音轨字幕（步骤 compose）：分段归一化拼接 → 铺 BGM/原声 → 烧字幕。

字幕按剧本台词生成 SRT/ASS；成片先过 detect_burned_text 检查画面里有没有模型自己带的字，
参考片带字幕时按 rules 的「字幕复刻」表决定要不要克隆参考片的字幕风格。
"""
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import shutil

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import produce_video  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from media import _ffmpeg, extract_audio
from shot_match import _shot_text
from task_store import _d, _p, _rel, log

from Agent_tools import registry as agent_tools  # noqa: E402


# ---------------- 步骤 8：拼接 + 音轨 + 字幕 ----------------
FONT_PATTERNS = [
    os.getenv("SUBTITLE_FONT", ""),
    "/usr/share/fonts/**/*CJK*.tt[cf]",
    "/usr/share/fonts/**/wqy*.tt[cf]",
    "/root/.comate-server/bin/*/extensions/baiducomate.comate/assets/font/HeiTi.ttf",
]


def _cjk_font() -> str:
    """找一个中文字体用于烧字幕，找不到返回空字符串（这时只产出 SRT）。"""
    for pattern in FONT_PATTERNS:
        if not pattern:
            continue
        if os.path.isfile(pattern):
            return pattern
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    return ""


_FONT_FAMILY: "dict[str, str]" = {}


def _font_family(path: str) -> str:
    """字体文件 → 字体族名（ASS 的 Fontname 认族名，不认文件名）。

    不能拿文件名 stem 当族名：NotoSansCJK-Black.ttc 的族名是「Noto Sans CJK SC」，
    写成 NotoSansCJK-Black 时 libass 匹配不到，会回落到默认西文字体再逐字找 fallback
    ——一行中文由 DejaVu/DroidSansJapanese/DroidSansFallback 几种字形拼出（字形不统一），
    在 fontconfig 里没注册 CJK 的机器上直接烧成方块，而产物里仍记着 burned=True。
    fc-query 读不到（没装 fontconfig）时退回 stem，至少不比现在差。
    .ttc 是字体集合，一个文件里好几个 face（Noto CJK 是 JP/KR/SC/TC/HK 五个），
    fc-query 会一行一个全列出来：中文字幕优先挑简体那个 face，
    排第一的 JP face 能显示汉字，但部分共用汉字是日文字形（「直」「骨」这类）。
    """
    if path in _FONT_FAMILY:
        return _FONT_FAMILY[path]
    stem = os.path.splitext(os.path.basename(path))[0]
    names = []
    try:
        ret = produce_video._run(["fc-query", "--format", r"%{family[0]}\n", path])
        if ret.returncode == 0:
            names = [x.strip() for x in (ret.stdout or "").splitlines() if x.strip()]
    except OSError:
        names = []
    sc = [x for x in names if "SC" in x.split() or "Simplified" in x or "SC" == x[-2:]]
    _FONT_FAMILY[path] = (sc or names or [stem])[0]
    return _FONT_FAMILY[path]


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms % 3600000 // 60000,
                                    ms % 60000 // 1000, ms % 1000)


# 字幕排版：全部按成片分辨率算，不依赖模型
SUB_FONT_SIZE = int(os.getenv("VF_SUB_FONT_SIZE", "42"))    # 以成片像素为单位
SUB_MARGIN_X = int(os.getenv("VF_SUB_MARGIN_X", "56"))      # 左右各留多少像素
SUB_MARGIN_V = int(os.getenv("VF_SUB_MARGIN_V", "72"))      # 距画面底部
SUB_MAX_LINES = 2                                            # 一屏最多两行，超了按时间切成多条


def _text_width(text: str) -> int:
    """按半宽单位估文本宽度：中日韩全角字符算 2，其余算 1。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _wrap_cue(text: str, per_line_units: int, max_lines: int = SUB_MAX_LINES) -> list:
    """把一条台词按半宽单位折行，返回若干「屏」，每屏最多 max_lines 行。

    纯字符宽度计算，不需要字体度量：中文字符宽度≈字号，所以每行能放
    (画面宽 - 左右边距) / 字号 个全角字，换算成半宽单位是它的 2 倍。
    行宽严格不超预算；靠近行尾遇到标点就提前断，英文回退到最近的空格断词。
    """
    breaks = "，。！？；：、,.!?;:"
    lines, cur = [], ""
    for ch in text.strip():
        if ch in " 　" and not cur:
            continue
        if _text_width(cur + ch) > per_line_units:
            head, rest = cur, ""
            if ch.isascii() and ch.strip() and " " in cur:
                left, _, tail = cur.rpartition(" ")   # 不能先 rstrip，否则丢掉词间空格
                if left.strip():                      # 英文别切在单词中间
                    head, rest = left, tail
            lines.append(head.strip())
            cur = rest + ch
        else:
            cur += ch
            if ch in breaks and _text_width(cur) >= per_line_units * 0.7:
                lines.append(cur.strip())
                cur = ""
    if cur.strip():
        lines.append(cur.strip())
    lines = [x for x in lines if x]
    if not lines:
        return []
    return ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]


def _pieces(built: dict, results: list) -> list:
    """成片时间轴上的「片」序列：[{"key","段号","镜头","file","url","start","end"}]。

    片 = 段内的一次裁剪块或一次 AI 补片（老产物没有「片」字段时整段算一片）。段内各片
    按实测时长按比例摊到这一段的实际长度上——归一化拼接会让时长有毫秒级出入，
    直接累加会一路漂下去。
    """
    out, cursor = [], 0.0
    for seg in built["分段"]:
        got = next((r for r in results if r["段号"] == seg["段号"] and r.get("file")), None)
        if not got:
            continue
        actual = produce_video._duration(got["file"])
        group = [p for p in (got.get("片") or []) if p.get("file")] or [
            {"file": got["file"], "url": got.get("url"), "镜头": seg.get("镜头序号") or []}]
        spans = [max(0.05, float(p.get("时长秒") or 0) or produce_video._duration(p["file"]))
                 for p in group]
        scale = (actual / sum(spans)) if sum(spans) else 1.0
        for i, (p, span) in enumerate(zip(group, spans), 1):
            width = span * scale
            out.append({"key": "%d#%d" % (seg["段号"], i), "段号": seg["段号"],
                        "镜头": list(p.get("镜头") or seg.get("镜头序号") or []),
                        "file": p["file"], "url": p.get("url") or "",
                        "start": cursor, "end": cursor + width})
            cursor += width
    return out


CAPTION_FX_KEYS = ("字幕", "花字", "文案", "标题", "贴纸")


def _ref_has_captions(rec: dict) -> bool:
    """参考片有没有叠加字幕/花字：读参考片拆解的分镜特效结论。

    只认「这一镜既标了字幕/花字，又确实有台词」的情况。片尾产品名板、logo 板这类画面
    内文字也会被拆解标成「字幕花字」，但它们没有台词——按关键词一命中就全片烧字幕，
    会让本来画面干净的成片凭空多一层字（17 号苹果笔记本实测：23 镜里只有片尾那一镜
    写着 "MacBook Neo"，台词是「无」，却把整片判成有字幕）。

    拆解漏标会导致该烧没烧，命令行可用 copy_subtitles="force" 显式打开兜底；
    读不到拆解结果时按 None 让规则表落到「烧」，宁可多字不静默丢信息。"""
    try:
        with open(_p(rec["task_id"], "reference", "analysis.json"), encoding="utf-8") as fh:
            ana = json.load(fh)
        for shot in (ana.get("分镜") or []):
            marked = any(any(k in str(fx) for k in CAPTION_FX_KEYS)
                         for fx in (shot.get("特效") or []))
            if marked and _shot_text(shot):
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def _dialog_timeline(built: dict, results: list) -> list:
    """按各片实际时长把剧本台词摊到全片时间轴：[(片key, start, end, text)]。

    text 是不折行的原始念白（分镜台词去掉「角色：」前缀）。基础排版（build_srt）和
    字幕风格克隆（_clone_ref_captions 的 items）共用这条时间轴，保证时间口径一致。
    摊到「片」而不是「段」：段内混合出片后，用户素材片是按分镜时长裁的、AI 补片是裁回
    计划时长的，按段平摊会把两种误差互相传染，字幕就对不上画面了。
    """
    shots = {s.get("序号"): s for s in built["剧本"].get("分镜") or []}
    rows = []
    for piece in _pieces(built, results):
        group = [shots.get(i, {}) for i in piece["镜头"]]
        width = piece["end"] - piece["start"]
        planned = sum(max(0.1, float(s.get("时长秒") or 0)) for s in group) or width
        cursor = piece["start"]
        for shot in group:
            span = width * max(0.1, float(shot.get("时长秒") or 0)) / planned
            # 无口播片没有台词，回退用剧本的「花字」文案，保证静音刷也有信息量
            text = _shot_text(shot) or str(shot.get("花字") or "").strip()
            if text:
                rows.append((piece["key"], cursor, cursor + span, text))
            cursor += span
    return rows


def build_srt(built: dict, results: list, dst: str, width: int = None,
              skip_keys: set = None) -> dict:
    """按台词时间轴写 SRT（基础排版链路）。

    width 给定时按画面宽度折行，保证烧上去不会超出画面；
    skip_keys 里的片跳过（这些画面自己已经带了字幕），但时间轴照常推进。
    返回 {"lines","skipped_shots","chars_per_line","cues"}，cues 供烧字幕时生成 ASS。
    """
    width = width or produce_video.TARGET_W
    per_line = max(8, int((width - 2 * SUB_MARGIN_X) / SUB_FONT_SIZE) * 2)  # 半宽单位
    skip = skip_keys or set()
    cues, skipped = [], 0
    for key, start, end, text in _dialog_timeline(built, results):
        if key in skip:
            skipped += 1
            continue
        screens = _wrap_cue(text, per_line) or [text]
        each = (end - start) / len(screens)
        for i, body in enumerate(screens):
            cues.append((start + i * each, start + (i + 1) * each, body))
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join("%d\n%s --> %s\n%s\n" % (i, _srt_time(a), _srt_time(b), t)
                           for i, (a, b, t) in enumerate(cues, 1)))
    return {"lines": len(cues), "skipped_shots": skipped, "chars_per_line": per_line // 2,
            "cues": cues}


def _apply_audio(rec: dict, src: str, dst: str) -> dict:
    """按 copy_bgm 与 step_audio 的判定结果处理成片音轨。

    成片人声一律是我们自己出的：voice_dub 先提音色基准，再用 TTS 克隆那把嗓子念我们的
    台词。所以这里绝不整轨照搬参考片原声——那会把克隆出来的口播和画面内音效全盖掉，
    成片听起来就成了原视频。参考片音轨只有 BGM 那一路可以贴回来。

    copy_bgm 管的是「复刻参考片的音乐」这一件事，管不到用户自己上传的 BGM：
    用户上传独立 BGM 是显式指定用哪首曲子（规则表「参考片音轨」第 1 行，优先级最高），
    关掉「复刻 BGM」不该把它一起丢掉。
    """
    opts, tid = rec["options"], rec["task_id"]
    plan_path = _p(tid, "audio", "reference_audio.json")
    plan = {}
    if os.path.isfile(plan_path):
        with open(plan_path, encoding="utf-8") as fh:
            plan = json.load(fh)

    user_bgm = plan.get("策略") == "用户上传音乐"
    if not opts["copy_bgm"] and not user_bgm:
        shutil.copyfile(src, dst)
        return {"mode": "keep_generated",
                "note": "保留各段生成音（克隆音色的口播与画面内音效）"}

    track = plan.get("bgm文件") or ""
    if not track or not os.path.isfile(track):
        shutil.copyfile(src, dst)
        return {"mode": "keep_generated", "strategy": plan.get("策略") or "未判定",
                "note": plan.get("原因") or "参考片没有可用 BGM，保留生成音"}
    # BGM 复刻口径（用户定）：不压低、原声直取。normalize=0 防 amix 把两路各砍半
    # （人声与 BGM 都保持原响度），限幅器只防两路叠加时的峰值爆音。
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-i", track,
           "-filter_complex",
           "[1:a]aloop=loop=-1:size=2e9[bg];"
           "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
           "alimiter=limit=0.95[a]",
           "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", dst]
    mode = "mix_bgm"
    note = "%s：%s，BGM 原声直取（不压低）" % (plan.get("策略"), plan.get("原因", ""))
    ret = produce_video._run(cmd)
    if ret.returncode != 0 or not os.path.isfile(dst):
        shutil.copyfile(src, dst)
        return {"mode": "keep_generated", "note": "音轨处理失败，保留生成音：%s" % ret.stderr[-200:]}
    return {"mode": mode, "note": note, "strategy": plan.get("策略"), "track": track}


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: %(w)d
PlayResY: %(h)d
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,%(font)s,%(size)d,&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,0,0,0,0,100,100,0,0,\
1,3,0,2,%(mx)d,%(mx)d,%(mv)d,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(sec: float) -> str:
    cs = int(round(sec * 100))
    return "%d:%02d:%02d.%02d" % (cs // 360000, cs % 360000 // 6000, cs % 6000 // 100, cs % 100)


def _fit_lines(text: str, per_line_units: int) -> str:
    """把每一行压到画面宽度以内。libass 的自动折行只在空格处断，中文没有空格
    根本不会折，所以最终宽度必须由我们自己保证。"""
    out = []
    for line in text.split("\n"):
        if _text_width(line) <= per_line_units:
            out.append(line)
        else:
            out.extend("\n".join(_wrap_cue(line, per_line_units, max_lines=99)).split("\n"))
    return "\n".join(x for x in out if x)


def _ass_body(text: str) -> str:
    """ASS 正文转义：先escape反斜杠与花括号，再把真实换行换成 \\N。

    `{...}` 在 ASS 里是样式 override block，台词里出现花括号时 libass 会把里面的内容
    连同括号一起吞掉；单独的反斜杠也会被当成控制码开头。台词是模型输出的，不可控。
    """
    out = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return out.replace("\n", "\\N")


def write_ass(cues: list, dst: str, width: int, height: int, font_name: str) -> str:
    """把 cue 列表写成 ASS。

    必须自己写 ASS 而不是让 ffmpeg 直接烧 SRT：SRT 没有分辨率信息，libass 按默认
    384x288 解释字号再拉伸到画面，字会被放大 4 倍多、直接冲出画面。ASS 里写明
    PlayResX/PlayResY = 成片实际分辨率，字号和边距才是真实像素。
    写入前再用 _fit_lines 兜一层，保证任何来源的文本都不会超出画面。
    """
    per_line = max(8, int((width - 2 * SUB_MARGIN_X) / SUB_FONT_SIZE) * 2)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(ASS_HEADER % {"w": width, "h": height, "font": font_name,
                               "size": SUB_FONT_SIZE, "mx": SUB_MARGIN_X, "mv": SUB_MARGIN_V})
        for start, end, text in cues:
            body = _ass_body(_fit_lines(text, per_line))
            fh.write("Dialogue: 0,%s,%s,Sub,,0,0,0,,%s\n"
                     % (_ass_time(start), _ass_time(end), body))
    return dst


def _burn_subtitles(cues: list, src: str, dst: str, ass_path: str) -> dict:
    font = _cjk_font()
    if not font:
        return {"burned": False, "note": "未找到中文字体，只产出 SRT（可设 SUBTITLE_FONT 指定字体）"}
    width, height = _video_size(src)
    family = _font_family(font)
    ass = write_ass(cues, ass_path, width, height, family)
    # 滤镜串里两个路径都要转义冒号，否则 ffmpeg 会把它当成参数分隔符（默认字体路径不带
    # 冒号所以现网不触发，但 SUBTITLE_FONT 是用户可配的）
    vf = "ass=%s:fontsdir=%s" % (ass.replace(":", r"\:"),
                                 os.path.dirname(font).replace(":", r"\:"))
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                             "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
                             dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        return {"burned": False, "note": "字幕烧制失败：%s" % ret.stderr[-200:]}
    return {"burned": True, "font": font, "font_family": family, "font_size": SUB_FONT_SIZE,
            "video_size": [width, height], "ass": ass}


def _video_size(path: str) -> tuple:
    """读画面宽高，读不到就按归一化目标值。"""
    err = produce_video._run([_ffmpeg(), "-hide_banner", "-i", path]).stderr
    for token in err.replace(",", " ").split():
        if "x" in token and token.replace("x", "").isdigit():
            w, h = token.split("x")
            if int(w) >= 64 and int(h) >= 64:
                return int(w), int(h)
    return produce_video.TARGET_W, produce_video.TARGET_H


TEXT_CHECK_PROMPT = """判断这段视频里有没有「叠加在画面上的字幕」，输出 json。

算字幕的：位于画面下方或中下方、跟着台词出现的成句文字；明显的花字、贴纸文案、水印。
不算字幕的：商品本身、包装、吊牌、衣服、招牌、书本、手机或电脑屏幕里印着或显示的文字，
这些是画面内容的一部分，不要算进来。

{"有字幕": true/false, "位置": "字幕出现在画面什么位置，没有写「无」",
 "内容": "字幕大致内容，没有写「无」"}
只输出 json。"""

# 这是个「有没有字幕」的二分判断，用最快的 VLM 就够，不必占用 gemini 的长上下文能力
SUBCHECK_ENGINE = os.getenv("VF_SUBCHECK_ENGINE", "qwen").strip()   # qwen | gemini
SUBCHECK_VERSION = 2      # 判定口径变了就 +1，旧缓存自动失效


def _has_subtitle(info: dict) -> bool:
    """兼容新旧字段名（有字幕 / 有画面文字）。"""
    value = info.get("有字幕")
    return bool(info.get("有画面文字")) if value is None else bool(value)


def _check_one_piece(piece: dict) -> dict:
    """检查一片画面里有没有叠加字幕。qwen 走公网 URL（seedance 产出的 URL 直接复用）。"""
    if SUBCHECK_ENGINE == "qwen":
        url = piece.get("url") or storage.upload(piece["file"])
        raw = aigc.vision(TEXT_CHECK_PROMPT, media=[{"type": "video", "url": url}],
                          max_tokens=1024, json_mode=True)
    else:
        raw = aigc.vision_gemini(TEXT_CHECK_PROMPT,
                                 media=[{"type": "video", "url": piece["file"]}])
    info = write_script._parse_json(raw)
    info["检测链路"] = SUBCHECK_ENGINE
    info["检测版本"] = SUBCHECK_VERSION
    return info


def _file_fingerprint(path: str) -> str:
    """片视频的内容指纹，用来判断缓存的检测结论还对不对得上这个文件。"""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_burned_text(rec: dict, pieces: list) -> dict:
    """检查每一片画面里是否自带字幕。自带的片跳过我们的字幕，避免叠字。

    按「片」而不是按「段」检测：段内混合出片后，一段里既有用户素材也有 AI 补片，
    按段判会因为一块自带字幕就让整段（含大量用户素材）都不叠字幕。
    结果缓存在 render/subtitle_check.json，重跑 compose 不会重复检测。
    缓存要同时对上判定版本与片视频的内容指纹：片被重新生成过就必须重新检测，
    否则会拿旧结论去判新画面（实测出现过段1叠字、段3该有字幕却被跳过）。
    """
    tid = rec["task_id"]
    path = _p(tid, "render", "subtitle_check.json")
    cache = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            cache = json.load(fh)

    def one(r):
        key = r["key"]
        fp = _file_fingerprint(r["file"])
        hit = cache.get(key) or {}
        if hit.get("检测版本") == SUBCHECK_VERSION and hit.get("片段指纹") == fp:
            return key, hit
        try:
            info = _check_one_piece(r)
        except Exception as exc:  # noqa: BLE001
            # 检测不了就按「没有字幕」处理：宁可我们贴字幕，也别整片没字幕
            info = {"有字幕": False, "检测失败": str(exc)[:150],
                    "检测链路": SUBCHECK_ENGINE, "检测版本": SUBCHECK_VERSION}
        info["片段指纹"] = fp
        return key, info

    todo = [r for r in pieces if r.get("file")]
    with cf.ThreadPoolExecutor(min(3, max(1, len(todo)))) as ex:
        for key, info in ex.map(one, todo):
            cache[key] = info
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)
    return cache


def _clone_ref_captions(rec: dict, built: dict, results: list, src: str,
                        skip: set) -> dict:
    """走 Agent_tools/caption_clone 复刻参考片字幕风格，返回工具的结果 dict。

    items 用台词时间轴（配音/剧本文本，比裸跑成片 ASR 准，不会有同音错字）；
    画面自带字幕的片不给 items，这些时间窗内不烧字，等效于基础排版的 skip。
    skip 的片还要作为 mute_spans 传下去：工具的「补漏」会把 plan 未覆盖的成片 ASR 念白填回来，
    不挡住就正好填在自带字幕上，成片会出现上下两行同一句话。
    """
    items = [{"start": round(a, 3), "end": round(b, 3), "text": t}
             for key, a, b, t in _dialog_timeline(built, results) if key not in skip]
    if not items:
        return {"ok": False, "error": "没有需要叠加的台词"}
    # rules.SUBTITLE_PLAIN 的口径翻成工具的环境变量：全片一种普通字幕，
    # 只复刻参考片的字体/字号/正文色，不复刻它的强调手段（关键词变色放大、斜排、顶部大字）。
    if rules.SUBTITLE_PLAIN:
        os.environ.update({"WHQ_CAPTION_PLAIN": "1", "WHQ_CAPTION_HOT_SCALE": "1.0",
                           "WHQ_CAPTION_POS": "bottom", "WHQ_CAPTION_ANIM": "off",
                           "WHQ_CAPTION_REVEAL": "block"})
    # 字幕文本只认剧本台词：补漏会把成片 ASR 没被 plan 覆盖的部分当念白补进来——
    # 有口播时是漏掉的台词，无口播时就是 BGM 歌词，后者烧出来是一屏乱码。
    # plan 已按分镜给了全部台词，不需要再从 ASR 补，关掉补漏。
    gapfill_prev = os.environ.get("WHQ_CAPTION_GAPFILL")
    os.environ["WHQ_CAPTION_GAPFILL"] = "0"
    try:
        return agent_tools.clone_captions(
            video=src, ref_video=rec["inputs"]["reference_video"],
            out_video=_p(rec["task_id"], "render", "with_capfx.mp4"),
            items=items, work_dir=_d(rec["task_id"], "render", "capwork"),
            mute_spans=[(p["start"], p["end"]) for p in _pieces(built, results)
                        if p["key"] in skip])
    finally:
        if gapfill_prev is None:
            os.environ.pop("WHQ_CAPTION_GAPFILL", None)
        else:
            os.environ["WHQ_CAPTION_GAPFILL"] = gapfill_prev


def step_compose(rec: dict) -> dict:
    tid = rec["task_id"]
    with open(_p(tid, "script", "script.json"), encoding="utf-8") as fh:
        built = json.load(fh)
    with open(_p(tid, "generated", "segments.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    files = [r["file"] for r in sorted(results, key=lambda r: r["段号"]) if r.get("file")]

    render = _d(tid, "render")
    concat = produce_video.concat(files, render)
    if concat.get("error"):
        raise RuntimeError(concat["error"])
    stitched = os.path.join(render, "stitched.mp4")
    os.replace(concat["file"], stitched)
    log(rec, "拼接 %d/%d 段完成" % (concat["segments_used"], len(results)))

    cur = os.path.join(render, "with_audio.mp4")
    audio = _apply_audio(rec, stitched, cur)
    log(rec, "音轨：%s" % audio["note"])

    # 字幕：按 rules「字幕复刻」表分叉（风格克隆 / 基础排版 / 不烧），判定痕迹进产物。
    # 前端开关只有两档：关掉=完全不烧；打开=跟随参考片——参考片画面本来就没字幕的，
    # 复刻出来也不该凭空多一层字。「强制烧」留给命令行 --subtitles（传字符串 "force"），
    # 以及老任务/老前端落盘的 True——它们的语义就是「用户显式要字幕」，
    # 不能跟默认的 None 混成「跟随参考片」，那会让显式打开的字幕一条都不烧。
    ref = rec["inputs"].get("reference_video") or ""
    opt = rec["options"].get("copy_subtitles")
    verdict = rules.decide("字幕复刻", {
        "字幕配置": ("关" if opt is False else "开" if opt in ("force", True) else "跟随参考片"),
        "参考片有字幕": _ref_has_captions(rec),
        "有参考片": bool(ref and os.path.isfile(ref)),
        "风格克隆可用": agent_tools.caption_available()})
    subs = {"enabled": verdict["动作"] != "不烧字幕", "判定": verdict}
    action, skip = verdict["动作"], set()
    # 不烧也要留痕：否则日志里既没有字幕行也没有原因，事后无法回答「为什么没字幕」
    log(rec, "字幕判定：%s（%s）" % (verdict["动作"], verdict.get("说明") or verdict.get("命中") or ""))
    # 没有口播台词就不烧字幕：字幕是台词的文字，没台词还烧只会把 BGM 歌词、画面花字
    # 当字幕塞进来（17 号苹果笔记本实测：无口播，字幕工具把 BGM 英文歌词识别成念白，
    # 烧出一屏重叠乱码）。
    if action != "不烧字幕" and not any(_shot_text(s)
            for s in (built["剧本"].get("分镜") or [])):
        action = "不烧字幕"
        subs.update({"enabled": False, "note": "参考片无口播，剧本没有台词，不生成字幕"})
        log(rec, "字幕：参考片无口播，没有台词可烧，跳过字幕")
    if action != "不烧字幕":
        pieces = _pieces(built, results)
        check = detect_burned_text(rec, pieces)
        keys = {p["key"] for p in pieces}     # 旧口径（按段）的缓存键不参与本次判定
        skip = {k for k, v in check.items() if k in keys and _has_subtitle(v)}
        subs["skipped_pieces"] = sorted(skip)
        if skip:
            log(rec, "片 %s 画面自带字幕，这些时间段不再叠加我们的字幕"
                % "、".join(sorted(skip)))
    if action == "风格克隆":
        got = _clone_ref_captions(rec, built, results, cur, skip)
        if got.get("ok"):
            cur = got["output"]
            subs.update({"mode": "风格克隆", "burned": True,
                         "blocks": got.get("blocks"), "lines": got.get("lines"),
                         "style": got.get("profile"),
                         # profile.styles 是参考片被读出来的样式档，不代表成片真烧了几种：
                         # 普通字幕口径下全部降级成口播档，这里显式记一笔免得看报告误会
                         "普通字幕": bool(rules.SUBTITLE_PLAIN)})
            if got.get("work_dir"):
                subs["work_dir"] = _rel(tid, got["work_dir"])
            log(rec, "字幕：风格克隆已烧入 %s 块 / %s 句（耗时 %.0fs）"
                % (got.get("blocks"), got.get("lines"), got.get("cost_s") or 0))
        else:
            action = verdict.get("回退") or "基础排版"
            subs["风格克隆失败"] = str(got.get("error"))[:200]
            log(rec, "字幕风格克隆失败，回退%s：%s" % (action, subs["风格克隆失败"]))
    if action == "基础排版":
        srt = _p(tid, "render", "subtitles.srt")
        info = build_srt(built, results, srt, width=_video_size(stitched)[0], skip_keys=skip)
        cues = info.pop("cues")
        subs.update(info)
        subs.update({"mode": "基础排版", "srt": _rel(tid, srt)})
        if cues:
            burned = os.path.join(render, "with_subs.mp4")
            subs.update(_burn_subtitles(cues, cur, burned, _p(tid, "render", "subtitles.ass")))
            if subs.get("burned"):
                cur = burned
        else:
            subs["note"] = "没有需要叠加的台词，未生成字幕"
        log(rec, "字幕：%s" % (("已烧入 %d 条，每行约 %d 字，跳过 %d 段"
                               % (info["lines"], info["chars_per_line"], len(skip)))
                               if subs.get("burned") else subs.get("note", "仅产出 SRT")))

    final = os.path.join(render, "final.mp4")
    if os.path.abspath(cur) != os.path.abspath(final):
        shutil.copyfile(cur, final)
    rec["result"] = {"final": _rel(tid, final),
                     "duration_sec": round(produce_video._duration(final), 1),
                     "audio": audio, "subtitles": subs,
                     "segments_used": concat["segments_used"], "segments_planned": len(results)}
    log(rec, "成片完成：%.1fs" % rec["result"]["duration_sec"])
    return {"artifact": rec["result"]["final"], "duration_sec": rec["result"]["duration_sec"]}
