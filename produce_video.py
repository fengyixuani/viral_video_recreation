"""端到端出片：参考爆款 + materials 下的商品素材 → 仿写剧本 → 分段生成视频 → 拼接成完整片。

跑法：python3 produce_video.py（要跑哪些编号、并发数都在 main() 里直接改）

每个编号（如 07）对应两份输入：
- output/script_analysis/{编号}_*.json —— 参考爆款的拆解结果（analyze_reference.py 产出）
- materials/{编号}/ 下的商品图 —— 文件名即商品名，卖点由 LLM 看图补全

流程：
1. 商品信息：VLM 读商品图 + LLM 按品类补出卖点/人群/文案，得到 write_script 需要的 product。
   理解类调用统一走 aigc.understand()，默认 gemini（config.UNDERSTAND_ENGINE / AIGC_ENGINE 可切）。
2. 仿写剧本 + 人物/场景设定图：直接复用 write_script.build()。
3. 分段生成：每段先按 Skills/SeedancePromptSkill.md（Seedance 提示词规范）把剧本的段提示词
   重写一遍（素材绑定 @图片N、镜头1/2/3 分镜、一镜一运镜、兜底约束包），再调
   line_art.gen_video_safe()，参考图 = 人物线稿设定图 + 商品原图。
   线稿参考图必须前置 REAL_ACTOR_HINT，否则成片里人物会顶着一张白色线稿脸。
4. 拼接：ffmpeg 先把各段归一化到同分辨率/帧率/音轨，再用 concat 复制流拼成 final.mp4。
   ffmpeg 取自 imageio_ffmpeg 自带的二进制，无需系统安装。

产物：output/produced/{编号}_{商品名}/ 下 segments/seg_*.mp4、final.mp4、produce.json
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import line_art  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(config.OUTPUT_DIR, "produced")
MATERIALS = os.path.join(HERE, "materials")
ANALYSIS = os.path.join(config.OUTPUT_DIR, "script_analysis")

def _canvas() -> "tuple[int, int]":
    """归一化画布跟随 config.ASPECT（生成画幅），短边 720，宽高取偶。
    写死 9:16 会把横屏参考片的复刻成片垫成上下黑边。"""
    try:
        w, h = (int(x) for x in str(config.ASPECT).split(":"))
    except (ValueError, AttributeError):
        w, h = 9, 16
    short = 720
    if w >= h:
        return (max(2, round(short * w / h / 2) * 2), short)
    return (short, max(2, round(short * h / w / 2) * 2))


TARGET_W, TARGET_H = _canvas()
TARGET_FPS = 30
# 音轨参数也必须统一：concat 是流复制，一条流里声道数中途从 mono 变 stereo 时，
# 后面再重编码这条流，AAC 编码器会在切换点吐「Input contains (near) NaN/+-Inf」而失败
# （实测：混了克隆配音片段(mono)和 seedance 补片(stereo)的段落会随机拼不进成片）。
TARGET_AR, TARGET_AC = 44100, 2


def aac_args(bitrate: str = "128k") -> list:
    """所有会进 concat 的片段都用这套音轨参数编码，见 TARGET_AR/TARGET_AC 的说明。"""
    return ["-c:a", "aac", "-b:a", bitrate, "-ar", str(TARGET_AR), "-ac", str(TARGET_AC)]


PRODUCT_INFO_PROMPT = """这是一张商品图，商品名是「__NAME__」。结合图片和商品名，输出 json：
{"name":"商品名","category":"品类",
 "selling_points":["3-4条卖点，具体可感知，别写空话"],
 "copy":"一句话广告语，10-15字",
 "audience":"目标人群",
 "appearance":"商品外观：外形结构、材质质感、主色配色、包装显著文字或图案，60字内",
 "usage_scene":"典型使用场景"}
卖点要符合这个品类的真实认知，不要编造检测数据或专利。只输出 json。"""


def _ffmpeg() -> str:
    """imageio_ffmpeg 自带的 ffmpeg 二进制，避免依赖系统安装。"""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# 单条 ffmpeg 的墙钟上限。给得很宽（长片重编码确实慢），只为兜死循环那种卡死，
# 不该把正常的长任务掐掉。用 VF_FFMPEG_TIMEOUT 可覆盖。
FFMPEG_TIMEOUT = config._env_int("VF_FFMPEG_TIMEOUT", 1800)


def _run(cmd: list, timeout: float = None) -> subprocess.CompletedProcess:
    """跑一条外部命令（基本都是 ffmpeg），返回码与 stderr 交给调用方判。

    默认带超时：线程池里某条 ffmpeg 卡住会永久占住 worker、整个任务不返回，
    而这类卡死是有实测记录的（apad 是无限流，配 -c:v copy 时 -shortest 收不住，
    实测跑 5 分钟不结束）。超时按「返回码非 0 + stderr 写明原因」返回，
    调用方原有的失败分支照旧生效，不用改。
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False,
                              timeout=timeout if timeout is not None else FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd, 124, stdout="",
            stderr="命令超时（%.0fs）未结束：%s" % (exc.timeout, " ".join(map(str, cmd))[:300]))


# ---------------- 输入配对 ----------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_inputs(tag: str) -> dict:
    """按编号找参考拆解 json 与商品图。找不到返回带 error 的 dict。"""
    refs = sorted(glob.glob(os.path.join(ANALYSIS, tag + "_*.json")))
    imgs = [p for p in sorted(glob.glob(os.path.join(MATERIALS, tag, "*")))
            if p.lower().endswith(IMAGE_EXTS)]
    if not refs:
        return {"tag": tag, "error": "缺少参考拆解 json：%s/%s_*.json" % (ANALYSIS, tag)}
    if not imgs:
        return {"tag": tag, "error": "materials/%s 下没有商品图" % tag}
    return {"tag": tag, "ref_json": refs[0], "images": imgs}


# ---------------- 第 1 步：商品信息 ----------------
def product_info(images: list) -> dict:
    """看图 + 按品类补全卖点，产出 write_script 需要的 product。"""
    main = images[0]
    name = os.path.splitext(os.path.basename(main))[0]
    url = storage.upload(main)
    # gemini 读本地图（base64 内联），回落链路才需要公网 URL；url 后面还要传给 seedance
    raw = aigc.understand(PRODUCT_INFO_PROMPT.replace("__NAME__", name),
                          media=[{"type": "image", "url": main}], max_tokens=2048,
                          json_mode=True)
    info = write_script._parse_json(raw)
    info.setdefault("name", name)
    info["images"] = images
    info["image_urls"] = [url] + [storage.upload(p) for p in images[1:]]
    return info


# ---------------- 第 3 步：提示词优化 + 分段生成 ----------------
SKILL_PATH = os.path.join(HERE, "Skills", "SeedancePromptSkill.md")

_skill_cache = None


def _skill_doc() -> str:
    """Seedance 2.0 提示词工程规范，整篇喂给 LLM 当重写依据。"""
    global _skill_cache
    if _skill_cache is None:
        with open(SKILL_PATH, encoding="utf-8") as fh:
            _skill_cache = fh.read()
    return _skill_cache


OPT_SYSTEM = "你是 Seedance 提示词优化专家。严格按给定规范重写提示词，输出 json。"

OPT_PROMPT = """下面是 Seedance 提示词规范，请严格按它把一段视频的提示词重写成可直接送模型的成品。

===== 规范全文 =====
__SKILL__
===== 规范结束 =====

【本段素材清单】按顺序对应 @图片1、@图片2 …：
__ASSETS__

【本段剧本】
__SEG__

【重写要求】
- 开头先用一句话交代整体设定（场景与光影走向、整体质感），随后按规范定义主体：
  「将 @图片N 中[2-3 个稳定静态特征] 定义为 <主体N>」，之后全程用 <主体N> 指代。
  人脸等需要精准参考的素材要尽量前置。
- 素材只能用 @图片N / <主体N> 指代，禁止裸写 URL 或 asset id；@图片N 紧接动词时补名词隔断。
- 本段含 __NSHOT__ 个镜头，用「镜头1 / 镜头2 …」推进，禁止写绝对秒数；
  每个镜头以一种运镜开头（如 固定 / 向前推 / 轻微摇摄），一个镜头只允许 1 种运镜。
- 每镜给了「转入转场」，必须把镜间怎么切写进这一镜的开头，不许自己另发明转场：
  硬切 / 无 → 写「硬切入本镜」，画面瞬间换掉，不要过渡、不要淡入淡出、不要转场动画；
  遮罩转场 / 划像 → 写清遮挡物（前景人影、手掌、衣物）怎么横穿入画、在遮住画面的瞬间完成切换；
  叠化 / 闪白闪黑 → 写成该效果本身；同镜续拍 / 运镜衔接 / 匹配剪辑 → 画面连续，不要切。
  换装、换色、换场景这类必须在切换的那一帧完成，切之前和切之后各自保持静止不变。
- 动作要肢体细化、程度量化，情绪用身体细节外化（眉头紧锁、手指颤抖）而不是抽象情绪词。
- 台词用 {}，画面内真实发生的物理音效用 <>（如 <炒菜滑动声>、<金属碰撞声>）。
  全片不使用背景音乐：不出现 （） 音乐标记，也不许把音乐伪装成音效——
  鼓点、节拍、电音、舞曲、弦乐、配乐、定音这类都属于音乐，一律不要写；
  剧本里提到的音乐起落一律忽略。<> 里的 <主体N> 是主体标签，照常使用。
- 全程正向描述，只说要什么：把「不要漫画感」写成「真人写实风格」，
  「人物面部不是线稿」写成「人物面部为真实自然的五官」。
- 删除剧本里所有字幕/花字/贴纸相关描述，画面里不出现文字。
- 商品必须清晰可辨，外观严格参考对应的商品图。
- 最后一段统一挂上一致性与画质约束：画面高清细节丰富、电影质感、色彩自然光影柔和；
  人物面部稳定不变形、五官清晰、动作连贯；人物身份服装发型配饰连续一致；
  场景空间关系与光影方向稳定；画面仅保留剧情设定中的人物；画面干净。

输出 json：
{"prompt":"重写后的完整提示词，可直接送给视频生成模型",
 "fixed":["自动补全或修正了哪些问题"],"principles":["套用了规范里的哪些原则"]}
只输出 json。"""

# 音乐一律由 compose 统一铺（参考片 BGM / 用户 BGM），段内只留画面内的物理音效：
# 多段拼接时各段音乐无法衔接，而写成 <> 的"鼓点音效"会被模型扩写成成段配乐（实测鼓点变弦乐）。
# 所以 （） 音乐标记全删，<> 只删音乐性的那些，<炒菜滑动声> 这类真实音效保留。
MUSIC_WORDS = ("bgm", "背景音乐", "音乐", "配乐", "乐曲", "鼓点", "节拍", "电音", "舞曲",
               "弦乐", "旋律", "和弦", "前奏", "副歌", "定音", "踩点", "卡点", "beat")
PAREN_MARK_RE = re.compile(r"（[^（）]*）|\([^()]*\)")
ANGLE_MARK_RE = re.compile(r"<[^<>]*>")
# <> 里这些是主体/道具标签，不是音频标记，必须保留
KEEP_MARK_RE = re.compile(r"^<(主体|道具|人物|角色|场景|物体)\s*\d*>$")


def _is_music(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in MUSIC_WORDS)


def scrub_music_marks(text: str) -> str:
    """删掉提示词里的音乐：（） 标记全删，<> 只删音乐性的。

    保留 <主体N> 这类标签、<炒菜滑动声> 这类画面内音效，以及台词 {}。
    """
    def angle(m):
        s = m.group(0)
        if KEEP_MARK_RE.match(s):
            return s
        return "" if _is_music(s) else s
    out = ANGLE_MARK_RE.sub(angle, PAREN_MARK_RE.sub("", text))
    lines = []
    for line in out.splitlines():
        # 标记删掉后常留下孤立的标点，逐行收拾干净
        line = re.sub(r"[ \t]{2,}", " ", line).replace("。。", "。").strip()
        lines.append(line.lstrip("。，、； "))
    return "\n".join(lines)


# 规范要求最关键的关键词前置，由代码拼在最前面，保证位置稳定
LEAD_NO_SUBTITLE = "保持无字幕。"
LEAD_NO_BGM = "无bgm。"


def _asset_manifest(char_urls: list, product_urls: list, assets: dict, product: dict,
                    state_notes: dict = None) -> str:
    """给 LLM 的素材清单，编号顺序必须与实际传给 seedance 的 ref_images 顺序一致。

    state_notes（url → {"关键状态","应有","应无"}）里的图是「中间状态图」：视频模型一次画不准的
    高精度状态先用图像编辑锁死，视频只负责状态之间的动态变化，所以清单里要单独说清。
    装配/拆解序列整条落在一块里时只会带起始/结束两张（note["位置"]），商品是故意不完整的：
    要明写演变方向、以及全段都不许出现的部件，否则模型会好心地一开始就把成品画出来。
    """
    cmap = {c.get("编号"): c for c in assets.get("人物") or []}
    notes = state_notes or {}
    lines, n = [], 0
    for cid in char_urls:
        n += 1
        c = cmap.get(cid) or {}
        lines.append("@图片%d = 人物形象设定图：%s（%s），全身正面，纯色背景"
                     % (n, c.get("姓名") or cid, cid))
    for u in product_urls:
        n += 1
        note = notes.get(u)
        if isinstance(note, str):        # 老任务的 shot_refs.json 里状态说明还是纯字符串
            note = {"关键状态": note}
        if not note:
            lines.append("@图片%d = 商品原图：%s。外观：%s"
                         % (n, product.get("name"), product.get("appearance") or "见图"))
            continue
        where = note.get("位置")
        if where:
            lines.append("@图片%d = 这一段的%s状态图（%s必须照它画）：%s"
                         % (n, where, "第一帧" if where == "起始" else "最后一帧",
                            note.get("关键状态") or ""))
        else:
            lines.append("@图片%d = 关键状态图（已锁定，必须照它画）：%s。"
                         "画面里商品的形态、状态、显示内容一律以这张图为准，"
                         "视频只做这个状态前后的动态变化，不要重新想象商品结构与状态"
                         % (n, note.get("关键状态") or ""))
        if where == "起始":
            lines.append("  这一段商品是%s的：从起始状态图的形态，一步步演变到结束状态图的形态。"
                         "必须从起始形态开始，不许一上来就是完整成品，也不许倒着演"
                         % ("逐步成形" if (note.get("叙事类型") or "装配") == "装配" else "逐步拆解"))
            if note.get("块内始终应无"):
                lines.append("  全段从头到尾都不能出现：%s" % "、".join(note["块内始终应无"]))
        elif not where and note.get("应无"):
            lines.append("  这一镜商品是故意不完整的：%s 尚未装上，画面里绝不能出现，"
                         "也不要提前补全或闪现，缺的位置就是平整表面"
                         % "、".join(note["应无"]))
    return "\n".join(lines) or "（本段无参考图，纯文生视频）"


def _sfx_only(text: str) -> str:
    """从剧本的「音效音乐」里只留画面内音效，按分句丢掉写音乐的部分。

    这个字段是音效和音乐混写的（如「入点炸起重低音鼓点，纯电子舞曲BGM起」），
    整条下发会诱导模型造配乐，整条丢掉又会失去「炒菜滑动声」这类有用信息。
    """
    parts = [p for p in re.split(r"[，,；;。\n]+", str(text or "")) if p.strip()]
    # 只留听得见的东西：分句得带「声/响/音」，否则是「情绪随画面收束」这类描述，不是音效
    keep = [p.strip() for p in parts
            if not _is_music(p) and re.search(r"[声响]|音效|音$", p)]
    return "、".join(keep) or "无"


def _seg_brief(seg: dict, shots: dict) -> str:
    """本段的剧本原文：段级提示词 + 段内每个镜头的编排字段。

    「转入转场」必须下发：卡点换装、遮罩转场这类爆款手法全在镜间衔接上，只给「镜头1/镜头2」
    的顺序，模型会自己补一段缓慢过渡或转场动画，换装的瞬切就没了。
    """
    out = []
    if seg.get("视觉锚点"):
        out.append("【全片视觉锚点】重写后必须原样保留这些元素的形态与颜色，不许改色改形："
                   + seg["视觉锚点"])
    out.append("整段画面：" + seg.get("视频提示词", ""))
    for i, idx in enumerate(seg.get("镜头序号") or []):
        s = shots.get(idx) or {}
        dialogue = str(s.get("台词") or "").strip()
        speech = ("人物必须在本镜中清晰说出这句台词，并生成与口型同步的自然人声：%s"
                  % dialogue) if dialogue else "本镜无台词，不要生成对白或人声"
        out.append("镜头%d（原剧本第%s镜）：转入转场=%s；景别=%s；运镜=%s；画面=%s；动作=%s；"
                   "台词=%s；对白要求=%s；画面内音效=%s；特效字幕=%s；叙事功能=%s"
                   % (i + 1, idx, s.get("转场") or "硬切", s.get("景别"), s.get("运镜"),
                      s.get("画面"), s.get("动作"), dialogue or "无", speech,
                      _sfx_only(s.get("音效音乐")),
                      "、".join(s.get("特效") or []) or "无", s.get("叙事功能")))
    return "\n".join(out)


def optimize_prompt(seg: dict, shots: dict, char_ids: list, product_urls: list,
                    assets: dict, product: dict, state_notes: dict = None) -> dict:
    """按 SeedancePromptSkill.md 重写本段提示词。失败则回退原始提示词。

    重写结果一律过 scrub_music_marks：规范正文与示例里满是 （音乐）/<鼓点音效>，
    模型很容易照抄，只靠指令压不住，所以出口再删一遍音乐（画面内音效保留）。
    """
    try:
        raw = aigc.understand(OPT_PROMPT
                              .replace("__SKILL__", _skill_doc())
                              .replace("__ASSETS__", _asset_manifest(char_ids, product_urls,
                                                                     assets, product, state_notes))
                              .replace("__SEG__", _seg_brief(seg, shots))
                              .replace("__NSHOT__",
                                       str(len(seg.get("镜头序号") or []))),
                              system=OPT_SYSTEM, max_tokens=8192, json_mode=True)
        rec = write_script._parse_json(raw)
        if not (rec.get("prompt") or "").strip():
            raise ValueError("优化后提示词为空")
        rec["prompt"] = scrub_music_marks(rec["prompt"])
        return rec
    except Exception as exc:  # noqa: BLE001
        return {"prompt": scrub_music_marks((seg.get("视觉锚点") or "") + seg["视频提示词"]),
                "optimize_error": str(exc)[:300]}


def gen_segment(seg: dict, assets: dict, product_urls: list, product: dict,
                shots: dict, outdir: str) -> dict:
    """生成一段视频。提示词先过 SeedancePromptSkill 优化，参考图 = 人物线稿设定图 + 商品原图。"""
    cmap = {c.get("编号"): c for c in assets.get("人物") or [] if c.get("url")}
    char_ids = [c for c in (seg.get("人物编号") or []) if c in cmap]
    refs = ([cmap[c]["url"] for c in char_ids] + product_urls)[:9]  # seedance 上限 9 张

    opt = optimize_prompt(seg, shots, char_ids, product_urls, assets, product)
    body = opt["prompt"].strip()
    for kw in (LEAD_NO_SUBTITLE, LEAD_NO_BGM):    # 模型常自己带上，去重避免重复开头
        while body.startswith(kw):
            body = body[len(kw):].lstrip()
    lead = LEAD_NO_SUBTITLE + LEAD_NO_BGM
    prompt = lead + (line_art.REAL_ACTOR_HINT if char_ids else "") + body

    rec = {"段号": seg["段号"], "生成时长秒": seg["生成时长秒"],
           "ref_images": refs, "prompt_final": prompt,
           "prompt_raw": seg["视频提示词"], "prompt_optimized": opt["prompt"],
           "prompt_fixed": opt.get("fixed"), "prompt_principles": opt.get("principles")}
    if opt.get("optimize_error"):
        rec["optimize_error"] = opt["optimize_error"]
    try:
        # keep_refs：商品图不参与线稿降级——线稿化只保人物外观，商品图会被改造成人物图
        out = line_art.gen_video_safe(prompt, ref_images=refs or None,
                                      keep_refs=product_urls,
                                      duration_sec=seg["生成时长秒"])
        rec["mode"] = out["mode"]
        rec["url"] = out["video_url"]
        rec["file"] = storage.download(out["video_url"],
                                       os.path.join(outdir, "seg_%02d.mp4" % seg["段号"]))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)[:400]
    return rec


# ---------------- 第 4 步：拼接 ----------------
def _has_audio(path: str) -> bool:
    return "Audio:" in _run([_ffmpeg(), "-hide_banner", "-i", path]).stderr


# 进 concat 的每一片都先把平均响度拉到同一档，再拼。不做这一步成片会有「换麦感」：
# TTS 配音的片干净、峰值留余量（9399 实测 -20.8dB / peak -4.5dB），用户原声的片带环境
# 底噪、峰值贴顶（-19.0dB / peak -0.7dB），19.5s 那一刀一听就是两条音轨接起来的。
# 只做静态增益 + 限幅：动态归一（loudnorm/compand）会把呼吸声和停顿一起抬起来。
TARGET_MEAN_DB = -20.0   # 目标平均响度（现有片子本来就落在 -19~-21，取中间值改动最小）
MAX_GAIN_DB = 6.0        # 单片最多抬/压多少：再多就是在把底噪当人声抬
MIN_GAIN_DB = 0.8        # 差不到这个值不值得动（听不出来，白改一次音轨）


def _audio_gain(src: str) -> float:
    """这一片要抬/压多少 dB 才对得齐 TARGET_MEAN_DB；静音片或量不出来返回 0。"""
    # 函数内导入：响度原语的唯一来源在 gates.py，而 gates 依赖本模块，模块级导入会成环
    import gates  # pyright: ignore[reportImplicitRelativeImport]
    db = gates.mean_db(src)
    if db <= gates.SILENT_DB:     # 本来就是静音片（分镜设计如此/补片没出声），不许抬
        return 0.0
    gain = TARGET_MEAN_DB - db
    if abs(gain) < MIN_GAIN_DB:
        return 0.0
    return max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain))


def _normalize(src: str, dst: str) -> bool:
    """统一分辨率/帧率/编码与响度，并保证一定有音轨（没有就补静音），concat 才能直接复制流。"""
    vf = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=%d" %
          (TARGET_W, TARGET_H, TARGET_W, TARGET_H, TARGET_FPS))
    cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    has_audio = _has_audio(src)
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-shortest"]
    cmd += ["-vf", vf]
    gain = _audio_gain(src) if has_audio else 0.0
    if gain:
        cmd += ["-af", "volume=%.2fdB,alimiter=limit=0.95" % gain]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"] + aac_args() + [dst]
    r = _run(cmd)
    return r.returncode == 0 and os.path.isfile(dst)


def concat(files: list, outdir: str) -> dict:
    """把各段拼成 final.mp4。返回 {"file"} 或 {"error"}。

    有片段归一化失败就整体报错：少一段的成片是静默内容丢失，比拼接失败更难发现。
    """
    if not files:
        return {"error": "没有可拼接的片段"}
    norm_dir = os.path.join(outdir, "normalized")
    os.makedirs(norm_dir, exist_ok=True)
    normed, bad = [], []
    for i, f in enumerate(files):
        dst = os.path.join(norm_dir, "n%02d.mp4" % i)
        if _normalize(f, dst):
            normed.append(dst)
        else:
            bad.append(os.path.basename(f))
    if bad:
        return {"error": "这些片段归一化失败，拼接中止（少一段的成片没有意义）：%s"
                         % "、".join(bad)}

    listfile = os.path.join(norm_dir, "list.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for p in normed:
            fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    final = os.path.join(outdir, "final.mp4")
    r = _run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
              "-safe", "0", "-i", listfile, "-c", "copy", "-movflags", "+faststart", final])
    if r.returncode != 0 or not os.path.isfile(final):
        return {"error": "concat 失败: %s" % r.stderr[-300:]}
    return {"file": final, "segments_used": len(normed)}


def _duration(path: str) -> float:
    """从 ffmpeg 输出里读时长，取不到返回 0。"""
    for line in _run([_ffmpeg(), "-hide_banner", "-i", path]).stderr.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            try:
                h, m, s = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except ValueError:
                return 0.0
    return 0.0


# ---------------- 单个编号的完整流程 ----------------
def produce(tag: str, seg_workers: int = 3, log=print) -> dict:
    src = find_inputs(tag)
    if src.get("error"):
        log("[%s] 跳过：%s" % (tag, src["error"]))
        return src

    log("[%s] 读商品图..." % tag)
    product = product_info(src["images"])
    name = "".join(c for c in product["name"] if c not in '/\\:*?"<>|')
    outdir = os.path.join(OUT_ROOT, "%s_%s" % (tag, name))
    segdir = os.path.join(outdir, "segments")
    os.makedirs(segdir, exist_ok=True)

    log("[%s] 仿写剧本 + 设定图（%s）..." % (tag, product["name"]))
    built = write_script.build(src["ref_json"], product, with_images=True)
    built.pop("outdir", None)
    with open(os.path.join(outdir, "script.json"), "w", encoding="utf-8") as fh:
        json.dump(built, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "script.md"), "w", encoding="utf-8") as fh:
        fh.write(write_script.to_markdown(built))

    segs = built["分段"]
    shots = {s.get("序号"): s for s in built["剧本"].get("分镜") or []}
    no_bgm = True  # 段内一律不要模型生成的音乐与音效，音轨统一由外部铺
    log("[%s] 优化提示词 + 生成 %d 段视频（禁用BGM与音效）..." % (tag, len(segs)))
    with cf.ThreadPoolExecutor(max(1, seg_workers)) as ex:
        results = list(ex.map(
            lambda s: gen_segment(s, built["素材"], product.get("image_urls") or [],
                                  product, shots, segdir),
            segs))
    for r in results:
        log("[%s]   段%d %s" % (tag, r["段号"],
                                r.get("mode") or ("FAIL " + r.get("error", "")[:80])))

    ok = [r["file"] for r in results if r.get("file")]
    log("[%s] 拼接 %d/%d 段..." % (tag, len(ok), len(segs)))
    final = concat(ok, outdir)

    rec = {"tag": tag, "product": product["name"], "reference": built["参考片"],
           "outdir": outdir, "shots": len(built["剧本"].get("分镜") or []),
           "planned_segments": len(segs), "generated_segments": len(ok), "no_bgm": no_bgm,
           "segment_results": results, "final": final}
    if final.get("file"):
        rec["final_duration_sec"] = round(_duration(final["file"]), 1)
        log("[%s] 完成 → %s（%.1fs）" % (tag, final["file"], rec["final_duration_sec"]))
    else:
        log("[%s] 拼接失败：%s" % (tag, final.get("error")))
    with open(os.path.join(outdir, "produce.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def main():
    # ==== 调试参数：直接改这里 ====
    TAGS = ["08"]        # 要出片的编号，对应 materials/{编号}
    TAG_WORKERS = 5      # 同时跑几个编号
    SEG_WORKERS = 5      # 每个编号内同时生成几段视频
    # ==============================

    os.makedirs(OUT_ROOT, exist_ok=True)
    lock = __import__("threading").Lock()

    def log(msg):
        with lock:
            print(msg, flush=True)

    with cf.ThreadPoolExecutor(max(1, TAG_WORKERS)) as ex:
        recs = list(ex.map(lambda t: produce(t, SEG_WORKERS, log), TAGS))

    print("\n==== 汇总 ====")
    for r in recs:
        if r.get("error"):
            print("%s  跳过：%s" % (r.get("tag"), r["error"]))
        else:
            f = r.get("final") or {}
            print("%s  %s  段 %d/%d  %s" % (
                r["tag"], r["product"], r["generated_segments"], r["planned_segments"],
                f.get("file") or ("失败：" + str(f.get("error"))[:80])))
    with open(os.path.join(OUT_ROOT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=2)
    print("汇总: %s" % os.path.join(OUT_ROOT, "summary.json"))
    return 0 if any((r.get("final") or {}).get("file") for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
