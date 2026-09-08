"""爆款复刻的判定规则集中在这里：规则是数据，加规则只加表里的一行。

跑法：``python3 rules.py``
产物：output/rules.md（所有规则表渲染成 markdown，用来通览规则逻辑），并做一次覆盖自检。

所有表共用一套语义（由 decide() 实现）：
- 自上而下取第一条命中的规则，每张表最后一行必须是无条件兜底行；
- 规则里不写某个字段 = 通配（写 None 同义）；
- 事实缺失（None）只能被通配吃掉，不会等值命中——漏采集的维度不会静默走进某条规则；
- 数值字段写 (下限, 上限)，含义是 下限 <= 值 < 上限；
- 「动作」只是动作名，实现由调用方的动作表（pipeline 里的 *_ACTIONS / 分支）映射到函数，
  表里只出现名字，换实现不用动表。

判定用到的事实（真人出镜口播 / 有BGM / 模特人脸出镜 …）由视频理解阶段直接给出：
素材侧见 analyze_materials.JUDGE_FIELDS，参考片侧见 analyze_reference.JUDGE_FIELDS，
不要在规则层再去猜关键词。
"""
import itertools
import json
import os
import sys

import config  # pyright: ignore[reportImplicitRelativeImport]

# ---------------- 表一：用户切片怎么用 ----------------
# 前置门槛：只看画面能不能用。口播内容对不上**不再**一票否决 —— 台词是新写的，用户素材里
# 说的几乎永远是别的话，卡这一条会把所有真素材淘汰掉；改成画面照用、声音重配（见下面第二行）。
# 兜底同理：画面过了门槛就一定要用上，音轨维度判不出来（素材标注缺 真人出镜口播/有BGM）时
# 按最安全的「静音 + 重新配音」处理，不能因为缺一条标注就把 0.8 匹配度的真素材整条丢掉
# （实测有一轮 12 镜全被兜底判成不使用，成片 0 帧用户素材）。
CLIP_GATES = ("画面匹配",)
CLIP_FIELDS = ("真人出镜口播", "口播内容匹配", "有BGM")

# 「真人出镜口播」的口径（与 analyze_materials 的标注定义一致）：画面里的人张嘴在说话
# （看得到嘴动、声音是他发出的）才算 True。人物出镜但没张嘴、只有旁白/画外音 → False，
# 命中第一行「静音后使用 + 重新配音」。
CLIP_RULES = [
    {"真人出镜口播": False,
     "动作": "静音后使用", "需要配音": True,
     "说明": "没有真人出镜口播：切片静音后直接用，口播交给配音"},
    {"真人出镜口播": True, "口播内容匹配": False,
     "动作": "静音后使用", "需要配音": True,
     "说明": "有人出镜说话但说的不是这句：画面照用，静音后重新配音"},
    {"真人出镜口播": True, "口播内容匹配": True, "有BGM": False,
     "动作": "原声直接使用", "需要配音": False,
     "说明": "口播内容也对得上、没有 BGM：原声直接用"},
    {"真人出镜口播": True, "口播内容匹配": True, "有BGM": True,
     "动作": "分离BGM保留人声", "需要配音": False,
     "说明": "口播内容对得上、但有 BGM：分离掉 BGM，仅保留有人声的切片"},
    {"动作": "静音后使用", "需要配音": True,
     "说明": "兜底：画面已过门槛、但音轨情况判不出来（标注缺失）→ 一律静音后使用 + 重新配音"},
]

# ---------------- 表二：多切片拼接的音色（有序，取第一个拿得到的）----------------
VOICE_PRIORITY = [
    {"来源": "用户视频-原声", "策略": "铆钉基准",
     "说明": "第一优先级：提取素材里的纯净人声（带 BGM 先分离）当整片音色基准"},
    {"来源": "AI视频片段", "策略": "参考原声",
     "说明": "素材没人声：第一个出声的 AI 段提纯人声当基准，给后续配音与 AI 段当参考音"},
    {"来源": "用户视频-静音", "策略": "克隆原声", "说明": "克隆基准音色后给静音切片配音"},
]

# ---------------- 表三：切片缺失，AIGC 补片怎么出模特 ----------------
GAP_FIELDS = ("需要模特出镜", "已有模特出镜", "多段需要模特")

GAP_RULES = [
    {"需要模特出镜": False,
     "动作": "AIGC直接生成", "依赖": ("商品图",),
     "说明": "这一段不需要人脸，直接生成"},
    {"需要模特出镜": True, "已有模特出镜": True,
     "动作": "提取模特改线稿图", "依赖": ("商品图", "视频模特帧"),
     "说明": "已选切片里有模特人脸：抽帧改线稿当参考图，人物锚到真实模特"},
    {"需要模特出镜": True, "已有模特出镜": False, "多段需要模特": True,
     "动作": "直接生成线稿图", "依赖": ("商品图", "视频模特帧"),
     "说明": "多段都要模特：先生成一张线稿模特帧，各段复用同一张保证一致"},
    {"需要模特出镜": True, "已有模特出镜": False, "多段需要模特": False,
     "动作": "文字描述生成", "依赖": ("商品图",),
     "说明": "只有一段要模特：不做人物参考图，靠文字描述生成"},
    {"动作": "AIGC直接生成", "依赖": ("商品图",),
     "说明": "兜底：判不出来就按不需要模特处理"},
]

# ---------------- 表四：分镜 ↔ 素材匹配度 → 出片策略 ----------------
DIRECT_USE_SCORE = 0.80      # ≥ 直接裁剪用户素材
EDIT_SCORE = 0.55            # ≥ 用素材做编辑生成，低于此宁可重新生成
MATCH_FIELDS = ("匹配度",)

# 一个素材片段只服务一个分镜（口径落地在 pipeline._assign_segments）。
# 允许跨分镜复用时，第二镜裁到的还是同一段画面（片段往往只有几秒），成片同画面播两遍。
# 所以：匹配度高的分镜先占，让位的分镜退到自己的次优候选，候选都被占才退回 AIGC 生成。
SEGMENT_EXCLUSIVE = True

# 一次生成最多带几张商品参考图。seedream/seedance 的参考图数量有限，且提示词里的
# @图片N 锚点必须与实际下发的参考图一一对应；商品图多于这个数时只带前几张当参考，
# 其余仍进商品事实卡（显式记日志，不静默丢）。
PRODUCT_REF_MAX = 9
PIECE_REF_MAX = 4    # 单个生成块最多带几张商品图：按分镜计划取并集后再截断，图越多模型越容易混搭

MATCH_RULES = [
    {"匹配度": (DIRECT_USE_SCORE, 1.01),
     "动作": "直接裁剪", "说明": "素材可直接用，裁剪对齐分镜时长"},
    {"匹配度": (EDIT_SCORE, DIRECT_USE_SCORE),
     "动作": "素材编辑", "说明": "画面可用但要改造，拿素材当参考视频编辑"},
    {"动作": "重新生成", "说明": "兜底：匹配度低于 %.2f，宁可重新生成" % EDIT_SCORE},
]

# ---------------- 表五：参考片音轨怎么复刻 ----------------
# 动作名与 _apply_audio 读的「策略」字段一致，改名要同步改那里。
REF_AUDIO_FIELDS = ("用户上传BGM", "参考片有BGM", "参考片有口播", "BGM是音乐")

# 「BGM是音乐」由 gates.check("BGM音乐性") 实测（声明与阈值见 gates.py 实测门禁范式）：
# False = 背景音只是零星音效或环境底噪，复刻它没有意义还会污染成片 → 不复用；
# None = 判定器判不出，按通配放行走原有整轨/分离逻辑，不因判定器挂掉丢 BGM。
REF_AUDIO_RULES = [
    {"用户上传BGM": True,
     "动作": "用户上传音乐", "说明": "用户上传了独立 BGM，优先使用"},
    {"用户上传BGM": False, "参考片有BGM": False,
     "动作": "不使用参考片音频", "说明": "参考片没有 BGM，有没有口播都不贴回来"},
    {"用户上传BGM": False, "参考片有BGM": True, "BGM是音乐": False,
     "动作": "不使用参考片音频",
     "说明": "参考片背景音只是零星音效/环境声（不是音乐），不当 BGM 复用"},
    {"用户上传BGM": False, "参考片有BGM": True, "参考片有口播": False,
     "动作": "整轨复用", "说明": "参考片只有 BGM 没有口播，整条音轨直接复用"},
    {"用户上传BGM": False, "参考片有BGM": True, "参考片有口播": True,
     "动作": "分离伴奏", "说明": "参考片同时有 BGM 和口播，只把分离出的伴奏贴回来"},
    {"动作": "不使用参考片音频", "说明": "兜底：判不出音轨内容就不贴参考片音频"},
]

# ---------------- 表六：字幕怎么烧 ----------------
# 动作由 pipeline.step_compose 的分支实现：
#   风格克隆 = Agent_tools/caption_clone 复刻参考片字幕风格（失败走「回退」动作）；
#   基础排版 = 自研 build_srt + ASS 烧录（白字底部居中，自己算折行）。
# 「风格克隆可用」= Agent_tools.registry.caption_available()，指工具本体可导入；
# 词级 ASR 缺失不算不可用（只是逐字时间退化，caption_clone 内部自己兜）。
# 「字幕配置」三值：开/关 是用户显式指定；「跟随参考片」是默认——复刻口径下，
# 参考片有字幕/花字才烧，参考片画面干净的成片也保持干净（17 号苹果广告复刻实测教训）。
SUBTITLE_FIELDS = ("字幕配置", "参考片有字幕", "有参考片", "风格克隆可用")

SUBTITLE_RULES = [
    {"字幕配置": "关",
     "动作": "不烧字幕", "说明": "用户明确关闭，成片不叠加字幕"},
    {"字幕配置": "跟随参考片", "参考片有字幕": False,
     "动作": "不烧字幕", "说明": "复刻口径：参考片画面干净无字，成片也不叠字"},
    {"有参考片": True, "风格克隆可用": True,
     "动作": "风格克隆", "回退": "基础排版",
     "说明": "用户开了字幕或参考片带字：复刻参考片字幕风格，失败回退基础排版"},
    {"动作": "基础排版", "说明": "没有参考片或克隆工具未就绪：走自研基础排版字幕"},
]

# 字幕长什么样：用户拍板「每个 case 统一风格，就是普通字幕，不要强调、不要高亮」。
# 所以风格克隆只复刻参考片的「字体/字号/颜色底色」，不复刻它的差异化手段——
# 块内关键词不变色不放大、不斜排、不加入场特效、位置一律底部居中。
# 落地：compose_video 调 caption_clone 前把这条口径翻成 WHQ_CAPTION_* 环境变量。
# 想临时看参考片原味花字，把 SUBTITLE_PLAIN 改 False（只影响观感，不影响时间轴与文本）。
SUBTITLE_PLAIN = True

# ---------------- 口播复刻 ----------------
# 成片要不要口播（台词+配音）由参考片事实决定：参考片纯 BGM/音效叙事的，
# 剧本一句台词都不许写——没有台词自然没有 TTS 配音、没有字幕，整条链自动闭环。
# 事实「参考片有人声口播」来自 analyze_reference 的整体分析；判不出来（None）按有口播处理。
VOICE_FIELDS = ("参考片有人声口播",)

VOICE_RULES = [
    {"参考片有人声口播": False,
     "动作": "不写台词",
     "说明": "复刻口径：参考片没有人声口播，成片纯画面+音效叙事，所有镜台词为空"},
    {"动作": "写台词", "说明": "参考片有口播（或判不出来按有处理）：照参考节奏写台词"},
]

TABLES = [
    {"名称": "用户切片处理", "字段": CLIP_FIELDS, "规则": CLIP_RULES,
     "门槛": CLIP_GATES, "门槛不过": "不使用",
     "说明": "每条命中素材的切片怎么处理（用/静音/分离BGM/不用）"},
    {"名称": "切片缺失补片", "字段": GAP_FIELDS, "规则": GAP_RULES,
     "说明": "没有可用切片、要 AIGC 补一段时，模特怎么出镜"},
    {"名称": "素材匹配策略", "字段": MATCH_FIELDS, "规则": MATCH_RULES,
     "说明": "分镜与素材片段的匹配度决定 直接裁剪 / 素材编辑 / 重新生成"},
    {"名称": "参考片音轨", "字段": REF_AUDIO_FIELDS, "规则": REF_AUDIO_RULES,
     "说明": "参考片音轨怎么复刻到成片"},
    {"名称": "字幕复刻", "字段": SUBTITLE_FIELDS, "规则": SUBTITLE_RULES,
     "说明": "成片字幕走参考片风格克隆、基础排版还是不烧"},
    {"名称": "口播复刻", "字段": VOICE_FIELDS, "规则": VOICE_RULES,
     "说明": "成片要不要口播（台词/配音）跟随参考片有没有人声口播"},
]

TABLE_BY_NAME = {t["名称"]: t for t in TABLES}


# ---------------- 事实收敛 ----------------
def as_bool(value) -> "bool":
    """把模型返回的 true/false/是/否/有/无 收敛成布尔；判不出来返回 None。

    规则表只把 None 当「未采集」，靠通配吃掉，绝不会等值命中某条规则。
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "是", "有", "yes", "y"):
        return True
    if text in ("false", "0", "否", "无", "没有", "not", "no", "n"):
        return False
    return None


# ---------------- 匹配器 ----------------
def _hit(rule: dict, fields: tuple, facts: dict) -> bool:
    for key in fields:
        want = rule.get(key)
        if want is None:                       # 通配
            continue
        got = facts.get(key)
        if isinstance(want, tuple):            # 数值区间 [下限, 上限)
            try:
                value = float(got)
            except (TypeError, ValueError):
                return False
            if not want[0] <= value < want[1]:
                return False
        elif isinstance(want, bool):
            if got is None or bool(got) is not want:
                return False
        elif got != want:
            return False
    return True


def decide(table: str, facts: dict) -> dict:
    """按规则表判定，返回命中行的输出字段 + 判定痕迹（规则表/命中行/事实）。

    痕迹会被调用方原样写进产物 json，前端和排查时能直接看出「这一条为什么这么处理」。

    前置门槛只认**明确的 False**。判不出来（None）照旧往下走规则，并在痕迹里记
    「门槛未判定」——门槛字段同样是模型标注，把 None 当 False 会让缺一条标注的
    0.9 匹配度素材被整条丢掉（表一注释里记着的那次事故：12 镜全被判成不使用、
    成片 0 帧用户素材），也和本模块「事实缺失只能被通配吃掉」的口径相反。
    """
    spec = TABLE_BY_NAME[table]                # 表名拼错就直接 KeyError，不要静默兜底
    unknown = []
    for gate in spec.get("门槛") or ():
        got = as_bool(facts.get(gate))
        if got is False:
            return {"动作": spec["门槛不过"], "说明": "前置门槛不过：%s" % gate,
                    "规则表": table, "命中行": -1, "事实": dict(facts)}
        if got is None:
            unknown.append(gate)
    for index, rule in enumerate(spec["规则"]):
        if _hit(rule, spec["字段"], facts):
            out = {k: v for k, v in rule.items() if k not in spec["字段"]}
            out.update({"规则表": table, "命中行": index, "事实": dict(facts)})
            if unknown:
                out["门槛未判定"] = list(unknown)
            return out
    raise RuntimeError("规则表「%s」缺兜底行" % table)


def pick_voice(available: "list") -> dict:
    """按表二取音色方案：available 是当前拿得到的来源名列表，取优先级最高的那个。"""
    for index, item in enumerate(VOICE_PRIORITY):
        if item["来源"] in available:
            return dict(item, 规则表="音色优先级", 命中行=index, 事实={"可用来源": list(available)})
    return {"来源": "", "策略": "全部AI配音", "说明": "兜底：没有任何用户原声可参考",
            "规则表": "音色优先级", "命中行": -1, "事实": {"可用来源": list(available)}}


# ---------------- 通览与自检 ----------------
def _cell(value) -> str:
    if value is None:
        return "任意"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], (int, float)):
        return "%.2f ≤ x < %.2f" % value
    if isinstance(value, (tuple, list)):
        return "、".join(str(v) for v in value)
    return str(value)


def _out_keys(rules: "list") -> "list":
    keys: "list" = []
    for rule in rules:
        for key in rule:
            if key not in keys:
                keys.append(key)
    return keys


def to_markdown() -> str:
    lines = ["# 复刻规则总表", "",
             "自上而下取第一条命中；「任意」= 通配；每张表最后一行是兜底行。", ""]
    for spec in TABLES:
        fields = list(spec["字段"])
        outs = [k for k in _out_keys(spec["规则"]) if k not in fields]
        lines += ["## %s" % spec["名称"], "", spec["说明"], ""]
        if spec.get("门槛"):
            lines += ["前置门槛：%s（任一不过 → %s）"
                      % ("、".join(spec["门槛"]), spec["门槛不过"]), ""]
        lines += ["| # | " + " | ".join(fields + outs) + " |",
                  "| --- | " + " | ".join("---" for _ in fields + outs) + " |"]
        for index, rule in enumerate(spec["规则"]):
            cells = [_cell(rule.get(k)) for k in fields] + [_cell(rule.get(k)) for k in outs]
            lines.append("| %d | %s |" % (index, " | ".join(cells)))
        lines.append("")
    lines += ["## 音色优先级（有序，取第一个拿得到的）", "",
              "| # | 来源 | 策略 | 说明 |", "| --- | --- | --- | --- |"]
    for index, item in enumerate(VOICE_PRIORITY):
        lines.append("| %d | %s | %s | %s |" % (index, item["来源"], item["策略"], item["说明"]))
    return "\n".join(lines) + "\n"


def _field_values(spec: dict, field: str) -> tuple:
    """这个字段在表里实际出现过的取值域，用于覆盖自检穷举。

    不能一律按 True/False 穷举：「字幕配置」的取值是 关/开/跟随参考片，
    穷举成布尔会报出一堆 {"字幕配置": true} 这种现实里不存在的假空洞
    （实测 12 条全是假的），把真的覆盖漏洞埋在噪音里，同时三个真实取值一次都没覆盖到。
    """
    seen = [r.get(field) for r in spec["规则"] if r.get(field) is not None]
    if seen and not all(isinstance(v, bool) for v in seen):
        return tuple(dict.fromkeys(seen))          # 保持表里出现的顺序，去重
    return (True, False)


def coverage_holes() -> dict:
    """穷举各字段取值的所有组合，报出只能落到兜底行的组合。

    加字段最容易出的问题是某些组合谁都不命中、静默走兜底，这里把它们列出来人工确认。
    """
    holes = {}
    for spec in TABLES:
        fields = [f for f in spec["字段"]
                  if all(not isinstance(r.get(f), tuple) for r in spec["规则"])]
        if len(fields) != len(spec["字段"]):
            continue                            # 含数值区间的表不穷举
        last = len(spec["规则"]) - 1
        miss = []
        for combo in itertools.product(*(_field_values(spec, f) for f in fields)):
            facts = dict(zip(fields, combo))
            facts.update({g: True for g in spec.get("门槛") or ()})
            if decide(spec["名称"], facts)["命中行"] == last:
                miss.append(facts)
        if miss:
            holes[spec["名称"]] = miss
    return holes


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "rules.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown())
    print("规则表 %d 张，音色优先级 %d 级" % (len(TABLES), len(VOICE_PRIORITY)))
    holes = coverage_holes()
    if holes:
        print("只能落到兜底行的组合（确认是否符合预期）：")
        for name, combos in holes.items():
            for facts in combos:
                print("  %s：%s" % (name, json.dumps(facts, ensure_ascii=False)))
    print("产物：%s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
