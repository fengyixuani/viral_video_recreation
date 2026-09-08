"""任务生成报告：把一次任务里每张图、每段视频的来龙去脉整理成一份 report.md。

回答三个问题：
1. 这张图 / 这段视频是用什么原料生成的（原图、参考视频、参考音、来源素材切片）
2. 喂给模型的最终 prompt 是什么（全文）
3. 产物落在哪、最后怎么被用进成片

数据全部来自任务目录里已有的中间产物 json，本模块只读不写（除 report.md 本身），
所以对任何历史任务、跑了一半失败的任务都能出报告。不 import pipeline（避免循环依赖），
字段全部按缺省容错：老任务缺新字段（如 白底图明细、参考视频来源）就按模板重建或标注缺失。

用法：
    python report.py <task_id>     # 手动给某个任务出报告
    python report.py               # 给最新一个任务出报告
流水线 pipeline.run() 每次跑完（无论成败）会自动调用 build()。
"""
import glob
import json
import os
import sys
import time

import config  # noqa: F401  先 import 它把 config.env 灌进环境，TASKS_DIR 才读得到 VF_TASKS_DIR
# 任务根目录必须跟 task_store 走同一份：Web 服务用 VF_TASKS_DIR 指到 output/web/tasks，
# 这里自己拼 output/tasks 的话所有 Web 任务都找不到目录，报告会静默出不来
# （pipeline._build_report 把异常吞成一行 print）。task_store 只依赖 stdlib + config，
# 不会把 pipeline 的循环依赖带进来。
from task_store import TASK_ROOTS, TASKS_DIR  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")

# 与 pipeline.WHITE_BG_PROMPT 保持一致，仅用于老任务（没记录 白底图明细）时重建 prompt 展示
WHITE_BG_TEMPLATE = ("纯白背景商品主图：%s。这一张要表现%s。"
                     "严格保留参考图中商品的外形结构、材质质感、配色与包装文字，不要改设计，"
                     "商品居中完整入画，影棚柔和均匀光，轻微接地阴影，"
                     "画面里只有商品本身，没有手、没有人、没有其它道具与背景陈设，4K 电商主图。")

MODE_NOTES = {
    "cut_user_video": "直接裁剪：画面就是用户素材原片，无 AI 生成",
    "edit_user_video": "素材编辑：把用户素材切片当参考视频，seedance 保留其镜头运动/构图，替换商品重演",
    "ref2v": "参考图生视频：无参考视频，靠参考图（商品图/人物设定图）+ 提示词生成",
    "line_art": "参考图生视频（线稿降级）：原参考图被真人风控拒，参考图转线稿后重试",
    "t2v": "纯文生视频：参考图全部被拒或没有，只按提示词生成",
    "mix_user_ai": "段内混合：用户素材裁剪片 + AI 补片拼接",
}


def _load(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _disp(tdir: str, p) -> str:
    """任务目录内的文件显示相对路径（报告在任务根目录，相对链接可点开），目录外保留原路径。"""
    if not p:
        return ""
    ap = os.path.abspath(str(p))
    td = os.path.abspath(tdir)
    return os.path.relpath(ap, td) if ap.startswith(td + os.sep) else str(p)


def _exists(tdir: str, p) -> bool:
    if not p:
        return False
    q = str(p)
    return os.path.isfile(q if os.path.isabs(q) else os.path.join(tdir, q))


def _img(tdir: str, p, alt: str = "") -> str:
    """图片文件给内嵌预览 + 路径；非图片或文件不存在只给路径。"""
    d = _disp(tdir, p)
    if not d:
        return "（无）"
    if os.path.splitext(d)[1].lower() in IMAGE_EXTS and _exists(tdir, p):
        return "![%s](%s)\n\n`%s`" % (alt or os.path.basename(d), d.replace(" ", "%20"), d)
    return "`%s`" % d


def _code(text) -> str:
    text = str(text or "").strip()
    return ("```text\n%s\n```" % text) if text else "（无）"


def _url_key(u: str) -> str:
    return str(u or "").split("?")[0]


def _sec_fmt(v) -> str:
    try:
        return "%.1fs" % float(v)
    except (TypeError, ValueError):
        return str(v or "")


def _same(flag) -> str:
    """商品一致性三态：True/False/None（比对失败或判不出）。"""
    return "一致" if flag is True else ("不一致" if flag is False else "判不出")


def _task_dir(task_id: str) -> str:
    """找任务目录。Web 与命令行任务分在两个根下（TASK_ROOTS），两边都翻一遍。
    都找不到就返回默认根下的路径，让 build() 报一个带真实路径的错。"""
    for root in TASK_ROOTS:
        path = os.path.join(root, task_id)
        if os.path.isdir(path):
            return path
    return os.path.join(TASKS_DIR, task_id)


class _Ctx:
    """一次报告用到的全部产物与索引。"""

    def __init__(self, task_id: str):
        self.tid = task_id
        self.tdir = _task_dir(task_id)
        j = lambda *p: os.path.join(self.tdir, *p)  # noqa: E731
        self.rec = _load(j("task.json")) or {}
        self.fact = _load(j("product", "fact_card.json")) or {}
        self.script = _load(j("script", "script.json")) or {}
        self.segments = _load(j("generated", "segments.json")) or []
        self.model_ref = _load(j("generated", "model_ref.json")) or {}
        self.shot_refs = _load(j("generated", "shot_refs.json")) or {}
        self.ref_audio = _load(j("audio", "reference_audio.json")) or {}
        self.voice = _load(j("audio", "voice_plan.json")) or {}
        self.matches = (_load(j("edit", "asset_matches.json")) or {}).get("分镜匹配") or []
        idx = _load(j("assets", "material_index.json")) or {}
        self.pool = {s.get("片段ID"): s for s in idx.get("片段") or []}
        self.url_names = self._build_url_names()

    def _build_url_names(self) -> dict:
        """公网 URL → (这张参考图是什么, 对应本地文件)。参考图在请求里只是一串 URL，
        报告要把它翻译回「商品图几 / 哪张设定图 / 音色基准」这种人能看懂的说法。"""
        names = {}
        imgs = self.fact.get("images") or []
        for i, u in enumerate(self.fact.get("image_urls") or []):
            names[_url_key(u)] = ("商品图%d（@图片%d）" % (i + 1, i + 1),
                                  imgs[i] if i < len(imgs) else "")
        assets = self.script.get("素材") or {}
        for c in assets.get("人物") or []:
            if c.get("url"):
                names[_url_key(c["url"])] = ("人物设定图 %s %s" % (c.get("编号") or "",
                                                                   c.get("姓名") or ""),
                                             c.get("file") or "")
        for c in assets.get("场景") or []:
            if c.get("url"):
                names[_url_key(c["url"])] = ("场景设定图 %s %s" % (c.get("编号") or "",
                                                                   c.get("名称") or ""),
                                             c.get("file") or "")
        if self.model_ref.get("url"):
            names[_url_key(self.model_ref["url"])] = (
                "模特线稿图（%s）" % (self.model_ref.get("说明") or "来源见 model_ref.json"), "")
        if self.voice.get("基准音URL"):
            names[_url_key(self.voice["基准音URL"])] = (
                "音色基准 voice_base.wav（%s）" % (self.voice.get("基准来源说明") or ""),
                self.voice.get("基准音文件") or "")
        return names

    def name_url(self, u: str) -> str:
        """把公网 URL 翻译成人类可读的「图几/设定图/音色基准」标签。"""
        got = self.url_names.get(_url_key(u))
        if not got:
            return "`%s`" % (str(u)[:110] + ("…" if len(str(u)) > 110 else ""))
        label, local = got
        return ("%s → `%s`" % (label, _disp(self.tdir, local))) if local else label


# ---------------- 各章节 ----------------
def _sec_head(ctx: _Ctx) -> list:
    rec, lines = ctx.rec, []
    lines += ["# 任务生成报告：%s" % (rec.get("title") or ctx.tid),
              "",
              "- 任务ID：`%s`" % ctx.tid,
              "- 状态：%s%s" % (rec.get("status") or "未知",
                                ("（停在 %s：%s）" % (rec.get("step"), (rec.get("error") or "")[:160]))
                                if rec.get("status") == "failed" else ""),
              "- 商品：%s（%s）" % (ctx.fact.get("name") or (rec.get("product") or {}).get("name")
                                    or "未命名", ctx.fact.get("category") or "品类未知"),
              "- 更新时间：%s，报告生成：%s" % (rec.get("updated_at") or "?",
                                               time.strftime("%Y-%m-%d %H:%M:%S"))]
    opts = rec.get("options") or {}
    if opts:
        lines += ["- 参数：" + "，".join("%s=%s" % (k, v) for k, v in opts.items())]
    steps = rec.get("steps") or {}
    if steps:
        lines += ["- 步骤：" + " → ".join("%s(%s)" % (k, (v or {}).get("status"))
                                          for k, v in steps.items())]
    return lines


def _sec_script_audit(ctx: _Ctx) -> list:
    """剧本层两道门禁的结论：逐镜的商品状态审查 + 整片的叙事连贯审查。

    整片这一层单独成节，是因为它回答的是别处都回答不了的问题——每一镜都合规、
    连起来却不知道在讲什么。盲测概要是核心证据：模型只看画面台词看出来的那句话，
    与「一句话概要」差得远就说明观众也看不出来。
    """
    sc = ctx.script.get("剧本") or {}
    audit, tale = sc.get("商品状态审查") or {}, sc.get("叙事连贯审查") or {}
    warns = sc.get("剧本告警") or []
    if not (audit or tale or warns):
        return []
    lines = ["", "## 二之二、剧本门禁", ""]
    if sc.get("一句话概要"):
        lines += ["- 剧本自述意图：%s" % sc["一句话概要"]]
    if tale:
        lines += ["", "### 整片叙事连贯审查（%s）" % (tale.get("状态") or "?"), "",
                  "- 可理解性：%s / 100（低于 %s 视为普通观众看不懂）"
                  % (tale.get("可理解性", "-"), 60),
                  "- 盲测概要（只看画面台词看出来的）：%s" % (tale.get("盲测概要") or "-"),
                  "- 意图兑现：%s" % (tale.get("意图兑现") or "-")]
        if tale.get("总体问题"):
            lines += ["- 主要问题：%s" % tale["总体问题"]]
        first, fix = tale.get("首轮") or {}, tale.get("修补") or {}
        if fix:
            lines += ["",
                      "- 自动修补：%s（%s）" % (fix.get("状态") or "?", fix.get("说明") or ""),
                      "- 首轮 %s 分 → 复审 %s 分%s"
                      % (first.get("可理解性", "-"),
                         (tale.get("复审") or {}).get("可理解性", "-"),
                         "，" + tale["复审说明"] if tale.get("复审说明") else "")]
            for d in fix.get("已改") or []:
                lines += ["", "  - 镜%s「%s」：%s" % (d.get("序号"), d.get("字段"),
                                                     d.get("理由") or ""),
                          "    - 原文：%s" % str(d.get("原文") or "")[:200],
                          "    - 改写：%s" % str(d.get("改写") or "")[:200]]
            for d in fix.get("未采纳") or []:
                lines += ["  - 未采纳 镜%s「%s」：%s" % (d.get("序号"), d.get("字段"),
                                                       d.get("未采纳原因") or "")]
        for key, head in (("理解断点", "理解断点（观众到这里跟不上）"),
                          ("孤立镜", "孤立镜（删掉不影响理解）"),
                          ("缺失交代", "缺失交代（相邻两镜之间少一环）"),
                          ("结构未兑现", "结构未兑现")):
            items = tale.get(key) or []
            if items:
                lines += ["", "%s：" % head, ""]
                lines += ["- %s" % json.dumps(x, ensure_ascii=False) for x in items]
    if audit:
        lines += ["", "### 逐镜商品状态审查（%s）" % (audit.get("状态") or "?"), ""]
        for v in audit.get("违规") or []:
            lines += ["- 镜%s %s（%s）：%s" % (v.get("序号"), v.get("类型") or "",
                                              v.get("字段") or "", v.get("问题") or "")]
        if audit.get("未修复"):
            lines += ["", "> 未修复 %d 处，会原样进生成端" % len(audit["未修复"])]
    if warns:
        lines += ["", "### 剧本告警", ""] + ["- %s" % w for w in warns]
    return lines


def _sec_inputs(ctx: _Ctx) -> list:
    ins = ctx.rec.get("inputs") or {}
    lines = ["", "## 一、输入素材", ""]
    if ins.get("reference_video"):
        lines += ["- 爆款参考片：`%s`" % ins["reference_video"]]
    for p in ins.get("user_videos") or []:
        lines += ["- 用户素材视频：`%s`" % p]
    for p in ins.get("product_images") or []:
        lines += ["- 用户商品图：`%s`" % p]
    for p in ins.get("person_images") or []:
        lines += ["- 人物参考图：`%s`" % p]
    if ins.get("bgm"):
        lines += ["- 用户 BGM：`%s`" % ins["bgm"]]
    return lines


def _rounds(ctx, rounds: list) -> list:
    """改造轮次明细：每轮 REFINE_BATCH 张候选、选中哪张、与原图比对结论。

    第 1 轮的指令在外面已经展示过，这里只展示后续轮次追加走样点后的指令。
    """
    out: list = []
    for r in rounds:
        comp = r.get("对比") or {}
        out += ["", "第 %s 轮（%d 张候选）：与原图比对 %s%s%s"
                % (r.get("轮次"), len(r.get("候选") or []), _same(comp.get("商品一致")),
                   "（%s）" % comp["差异"] if comp.get("差异") else "",
                   "，选中理由：%s" % comp["理由"] if comp.get("理由") else "")]
        if r.get("轮次") != 1 and r.get("prompt"):
            out += ["", "本轮指令（追加了上一轮的走样点）：", _code(r.get("prompt"))]
        for k, c in enumerate(r.get("候选") or [], 1):
            if not c.get("文件"):
                out += ["", "- 候选 %d：生成失败 %s" % (k, c.get("error") or "")]
                continue
            mark = "（选中）" if c.get("文件") == r.get("最佳") else ""
            out += ["", "候选 %d%s：" % (k, mark), "", _img(ctx.tdir, c.get("文件")), ""]
    return out


def _sec_product_images(ctx: _Ctx) -> list:
    """商品参考图：每张的来源（用户上传 / 素材抽帧+白底生成），抽帧与白底化全过程。"""
    lines = ["", "## 二、商品参考图（后续生图生视频的 @图片N 就是它们）", ""]
    imgs = ctx.fact.get("images") or []
    harvest = ctx.fact.get("商品图补齐") or {}
    src_note = ctx.fact.get("商品图来源") or ("用户素材抽帧 + 白底图生成" if harvest else "用户上传")
    lines += ["共 %d 张，来源：%s。" % (len(imgs), src_note), ""]
    for i, p in enumerate(imgs, 1):
        lines += ["### 商品图%d（提示词里的 @图片%d）" % (i, i), "", _img(ctx.tdir, p), ""]

    idx = ctx.fact.get("商品图索引") or []
    if idx:
        lines += ["### 商家图片素材的结构化理解（选图与编辑判断都读这份）", "",
                  "| 编号 | 文件 | caption | 视角/状态 | 完整性 | 局部特写 | 商品占比 | 干扰 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for d in idx:
            dirty = [str(x) for x in (d.get("干扰") or []) if str(x).strip()]
            lines += ["| %s | `%s` | %s | %s | %s | %s | %s | %s |"
                      % (d.get("编号"), _disp(ctx.tdir, d.get("文件")),
                         (d.get("caption") or "").replace("|", "/"),
                         d.get("视角或状态") or "?", d.get("完整性") or "?",
                         "是" if d.get("是否局部细节") else "否", d.get("商品占比") or "?",
                         "、".join(dirty).replace("|", "/") or "无")]
        shows = [d for d in idx if d.get("展示信息")]
        if shows:
            lines += [""] + ["- 编号 %s 展示信息：%s"
                             % (d.get("编号"), "、".join(str(x) for x in d.get("展示信息") or []))
                             for d in shows]
        lines += [""]

    sel = ctx.fact.get("商品图选图") or {}
    if sel:
        lines += ["### 参考图选图（只读结构化理解，不再看图）", ""]
        if sel.get("说明"):
            lines += ["- %s" % sel["说明"], ""]
        if sel.get("主锚点换位"):
            lines += ["- 主锚点换位（多款同框的图不能当 @图片1）：%s" % sel["主锚点换位"], ""]
        for i, c in enumerate(sel.get("选中") or [], 1):
            lines += ["- @图片%d ← 编号 %s `%s`：%s（%s）"
                      % (i, c.get("编号"), _disp(ctx.tdir, c.get("文件")),
                         c.get("角色") or "", c.get("理由") or "")]
        for c in sel.get("落选") or []:
            lines += ["- 落选 编号 %s `%s`：%s"
                      % (c.get("编号"), _disp(ctx.tdir, c.get("文件")), c.get("原因") or "")]
        if sel.get("缺失"):
            lines += ["- 缺失（素材里没有，按规则不虚构）：%s"
                      % "、".join(str(x) for x in sel["缺失"])]
        lines += [""]

    checks = ctx.fact.get("商品图检查") or []
    if checks:
        lines += ["### 选中图的编辑判断（VLM 一次写好编辑指令 → 一张一张生成 → 每张与原图比对）", ""]
        for i, c in enumerate(checks, 1):
            v = c.get("判定") or {}
            lines += ["**第 %d 张** `%s`" % (i, c.get("原图")), "",
                      "- 判定：可直接用=%s，商品=%s，问题：%s%s"
                      % (v.get("可直接用"), v.get("商品") or "?", v.get("问题") or "无",
                         "（依据 %s）" % v["依据"] if v.get("依据") else ""),
                      "- 结论：%s" % (c.get("结论") or "")]
            if c.get("prompt"):
                lines += ["", "编辑指令（%s）：" % ("模型没给指令，按兜底模板"
                                                   if c.get("指令兜底") else "VLM 看图一次写好"),
                          _code(c.get("prompt"))]
            lines += _rounds(ctx, c.get("轮次") or [])
            if c.get("最终") and c.get("最终") != c.get("原图"):
                lines += ["", "原图 → 最终采用：", "", _img(ctx.tdir, c.get("原图")), "",
                          _img(ctx.tdir, c.get("最终")), ""]
            lines += [""]
    vet = (ctx.shot_refs or {}).get("选中补体检") or []
    if vet:
        lines += ["### 分镜级选图的补体检（这些图不在全局参考图里，被某一镜选中才体检）", ""]
        for c in vet:
            v = c.get("判定") or {}
            lines += ["**编号 %s** `%s`" % (c.get("编号"), c.get("原图")), "",
                      "- 判定：可直接用=%s，问题：%s" % (v.get("可直接用"), v.get("问题") or "无"),
                      "- 结论：%s" % (c.get("结论") or "")]
            if c.get("prompt"):
                lines += ["", "编辑指令：", _code(c.get("prompt"))]
            lines += _rounds(ctx, c.get("轮次") or [])
            if c.get("最终") and c.get("最终") != c.get("原图"):
                lines += ["", "原图 → 最终采用：", "", _img(ctx.tdir, c.get("原图")), "",
                          _img(ctx.tdir, c.get("最终")), ""]
            lines += [""]
    if not harvest:
        return lines

    lines += ["### 抽帧过程（用户没传商品图，从素材里补齐）", ""]
    for c in harvest.get("候选") or []:
        lines += ["- 片段 `%s`：从 `%s` 第 %s 秒抽帧 → `%s`（%s）"
                  % (c.get("片段ID"), c.get("源文件"), c.get("抽帧秒数"),
                     c.get("文件"), c.get("理由") or "")]
    judge = harvest.get("判定") or {}
    if judge:
        lines += ["", "最佳帧判定：清晰=%s，问题=%s，帧=`%s`"
                  % (judge.get("清晰"), judge.get("问题") or "无", judge.get("帧文件") or "")]

    details = harvest.get("白底图明细") or []
    if details:
        lines += ["", "### 白底图生成（seedream 图生图，原帧进 ref_images；一张一张生成，每张与源帧比对）", ""]
        for i, d in enumerate(details, 1):
            lines += ["**白底图 %d**：源帧 `%s`%s，最终采用："
                      % (i, d.get("源帧"), "（已弃用，不进参考图）" if d.get("弃用") else ""), "",
                      _img(ctx.tdir, d.get("文件")), "", "prompt：", _code(d.get("prompt")), ""]
            if d.get("轮次"):
                lines += _rounds(ctx, d.get("轮次") or []) + [""]
            else:       # 老任务只记了一次比对结论
                comp = d.get("原图对比") or {}
                if comp:
                    lines += ["与源帧比对：%s%s" % (_same(comp.get("商品一致")),
                              "（%s）" % comp["差异"] if comp.get("差异") else ""), ""]
    elif harvest.get("白底图"):
        # 老任务没记录明细：prompt 按当时的模板重建展示
        desc = judge.get("商品") or ctx.fact.get("name") or "该商品"
        # 元素可能是裸字符串（模型把单字段对象简写了）或 null：报告只是展示，
        # 不能因为这个把整份报告的生成弄崩。product_images 那边有同样的归一。
        wants = [(w.get("要表现什么") if isinstance(w, dict) else w) or "商品正面全貌"
                 for w in (judge.get("白底图") or [{}])]
        lines += ["", "### 白底图生成（seedream 图生图，原帧进 ref_images；prompt 按模板重建）", ""]
        for i, p in enumerate(harvest.get("白底图") or [], 1):
            want = wants[i - 1] if i <= len(wants) else "商品正面全貌"
            lines += ["**白底图 %d**：源帧 `%s` → 产物：" % (i, judge.get("帧文件") or ""), "",
                      _img(ctx.tdir, p), "", "prompt（重建）：",
                      _code(WHITE_BG_TEMPLATE % (desc, want)), ""]
    return lines


def _sec_assets(ctx: _Ctx) -> list:
    """人物/场景设定图 + 模特线稿图：原料、prompt、产物。"""
    assets = ctx.script.get("素材") or {}
    if not (assets.get("人物") or assets.get("场景") or ctx.model_ref.get("url")):
        return []
    lines = ["", "## 三、设定图（写剧本时生成，供 AI 补片当人物/场景锚点）", ""]
    for kind, key_name in (("人物", "姓名"), ("场景", "名称")):
        for c in assets.get(kind) or []:
            title = "%s设定图 %s %s" % (kind, c.get("编号") or "", c.get(key_name) or "")
            lines += ["### %s" % title.strip(), "",
                      "- 生成方式：%s%s" % (c.get("mode") or "?",
                                            "（人物图走线稿链路是为了过 seedance 真人风控）"
                                            if kind == "人物" else "")]
            refs = c.get("ref_images") or []
            if refs:
                lines += ["- 参考图（原料）："] + ["  - %s" % ctx.name_url(u) for u in refs]
            else:
                lines += ["- 参考图（原料）：无，纯文生图"]
            if c.get("error"):
                lines += ["- 生成失败：%s" % c["error"]]
            lines += ["", "prompt：", _code(c.get("提示词")), "",
                      "产物：", "", _img(ctx.tdir, c.get("file")), ""]
    if ctx.model_ref.get("url") or ctx.model_ref.get("说明"):
        lines += ["### 模特线稿图（从选中的用户切片抽帧 → 人物外观转文字 → 线稿重绘）", "",
                  "- %s" % (ctx.model_ref.get("说明") or "未生成"),
                  "- 产物 URL：`%s`" % (ctx.model_ref.get("url") or "无"),
                  "- 中间帧：`generated/model_frame.jpg`（如存在）", ""]
    return lines


def _sec_match(ctx: _Ctx) -> list:
    """分镜匹配表：解释每一镜为什么走素材 / 素材编辑 / 重新生成。"""
    if not ctx.matches:
        return []
    lines = ["", "## 四、分镜 × 用户素材匹配（决定每一镜怎么来）", "",
             "| 镜头 | 计划时长 | 策略 | 素材片段 | 匹配度 | 理由 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in ctx.matches:
        lines += ["| %s | %s | %s | %s | %.2f | %s |"
                  % (r.get("序号"), _sec_fmt(r.get("时长秒")), r.get("策略") or "",
                     r.get("片段ID") or "—", float(r.get("匹配度") or 0),
                     str(r.get("理由") or "").replace("|", "、")[:60])]
    return lines


def _piece_cut(ctx: _Ctx, p: dict) -> list:
    lines = ["- 生成方式：%s" % MODE_NOTES["cut_user_video"]]
    for d in p.get("片段") or []:
        seg_meta = ctx.pool.get(d.get("片段ID")) or {}
        lines += ["  - 镜%s ← 片段 `%s`（`%s` 的 %s~%s 秒，速度 %s，匹配度 %.2f，%s）"
                  % (d.get("镜头序号"), d.get("片段ID"),
                     seg_meta.get("源文件") or "源文件见素材索引",
                     (d.get("源窗口") or ["?", "?"])[0], (d.get("源窗口") or ["?", "?"])[-1],
                     d.get("speed"), float(d.get("匹配度") or 0),
                     d.get("切片动作") or "")]
    return lines


def _piece_gen(ctx: _Ctx, p: dict) -> list:
    mode = p.get("mode") or "?"
    lines = ["- 生成方式：%s —— %s" % (mode, MODE_NOTES.get(mode, ""))]
    gap = p.get("补片判定") or {}
    if gap:
        lines += ["- 补片判定：%s（%s）" % (gap.get("动作") or "", gap.get("说明") or "")]
    src = p.get("参考视频来源") or {}
    local_clip = p.get("参考视频文件") or ""
    if not local_clip and p.get("ref_videos"):
        # 旧任务没记录参考视频来源：按命名约定从产物名推回 generated/clips/ref{tag}.mp4
        base = os.path.basename(p.get("file") or "")
        tag = base[4:-4] if base.startswith("seg_") and base.endswith(".mp4") else ""
        cand = os.path.join(ctx.tdir, "generated", "clips", "ref%s.mp4" % tag)
        local_clip = cand if tag and os.path.isfile(cand) else ""
    for u in p.get("ref_videos") or []:
        note = ("剪自片段 `%s`（`%s` 镜%s，匹配度 %.2f，%s）"
                % (src.get("片段ID"), src.get("源文件"), src.get("镜头"),
                   float(src.get("匹配度") or 0), _sec_fmt(src.get("时长秒")))
                if src else "剪自匹配度达标的用户素材切片（来源细节见运行日志）")
        lines += ["- 参考视频 @视频1：`%s`，%s" % (_disp(ctx.tdir, local_clip) or "本地文件未存档",
                                                   note),
                  "  - 上传后 URL：`%s`" % (_url_key(u)[:110])]
    refs = p.get("ref_images") or []
    if refs:
        lines += ["- 参考图（按顺序）："] + ["  - %d. %s" % (i, ctx.name_url(u))
                                            for i, u in enumerate(refs, 1)]
    for c in p.get("商品图计划") or []:
        lines += ["- 镜%s 的商品图由剧本决定：商品图%s（%s）"
                  % (c.get("镜头"),
                     "、".join(str(x) for x in (c.get("参考图编号") or [])) or "不带商品图",
                     c.get("理由") or "")]
        mid = c.get("中间图") or {}
        if not (c.get("关键状态") or mid):
            continue
        lines += ["  - 关键状态（视频模型一次画不准，先用图像编辑锁死）：%s"
                  % (c.get("关键状态") or mid.get("关键状态") or ""),
                  "  - 中间状态图：%s" % (mid.get("来源") or "未生成")]
        man = mid.get("装配清单") or {}
        if man:
            lines += ["  - %s序列第 %s/%s 步，基于商品图%s 减部件生成；应有：%s；应无：%s"
                      % (man.get("叙事类型") or "装配", man.get("序号"), man.get("共几步"),
                         man.get("基准编号"), "、".join(man.get("应有") or []),
                         "、".join(man.get("应无") or []) or "（无）")]
        if mid.get("prompt"):
            lines += ["", "  编辑指令（基于商品图%s）：" % mid.get("基准编号"),
                      _code(mid.get("prompt")), ""]
        if mid.get("文件"):
            lines += ["", _img(ctx.tdir, mid.get("文件")), ""]
        if mid.get("轮次"):
            lines += _rounds(ctx, mid.get("轮次") or []) + [""]
    used = p.get("ref_images_used")
    if used and [_url_key(u) for u in used] != [_url_key(u) for u in refs]:
        lines += ["- 实际下发参考图（真人风控降级后人物图已换成线稿版，商品图保持原样）："]
        lines += ["  - %d. %s" % (i, ctx.name_url(u)) for i, u in enumerate(used, 1)]
    elif mode == "line_art" and used is None:
        lines += ["- 注意：这条产物出自旧版降级逻辑，被风控拒后所有参考图（含商品图）"
                  "都被线稿链路重绘成人物图再下发，商品外观只能靠提示词文字还原；"
                  "新版已改为只线稿化人物图、商品图保持原样。"]
    for u in p.get("ref_audios") or []:
        lines += ["- 参考音（让 AI 段说话保持同一把嗓子）：%s" % ctx.name_url(u)]
    return lines


def _sec_segments(ctx: _Ctx) -> list:
    if not ctx.segments:
        return []
    lines = ["", "## 五、分段视频（成片的每一段怎么来的）", ""]
    miss = (ctx.shot_refs or {}).get("缺失") or []
    if miss:
        lines += ["剧本按镜要商品图时缺的角度（素材里没有，不虚构）：", ""] + \
                 ["- 镜%s 需要 %s：%s" % (m.get("镜头"), m.get("需要") or "", m.get("说明") or "")
                  for m in miss] + [""]
    for seg in ctx.segments:
        lines += ["### 段 %s：%s" % (seg.get("段号"),
                                     MODE_NOTES.get(seg.get("mode") or "", seg.get("mode") or "")),
                  "",
                  "- 产物：`%s`（%s）" % (_disp(ctx.tdir, seg.get("file")),
                                          _sec_fmt(seg.get("生成时长秒")))]
        if seg.get("error"):
            lines += ["- 失败：%s" % seg["error"]]
        if seg.get("用户素材镜头"):
            lines += ["- 用户素材镜头：%s" % seg["用户素材镜头"]]
        if seg.get("AI补片镜头"):
            lines += ["- AI 补片镜头：%s" % seg["AI补片镜头"]]
        for p in seg.get("片") or []:
            lines += ["", "#### %s（镜头 %s，%s）" % (p.get("label") or "片", p.get("镜头"),
                                                      _sec_fmt(p.get("时长秒"))), ""]
            lines += _piece_cut(ctx, p) if p.get("kind") == "cut" else _piece_gen(ctx, p)
            for k in ("配音", "念白", "配音变速", "配音截断秒", "补配音", "降级", "error"):
                if p.get(k):
                    lines += ["- %s：%s" % (k, p[k])]
            lines += ["- 产物：`%s`" % _disp(ctx.tdir, p.get("file"))]
            if p.get("kind") != "cut":
                lines += ["", "最终提示词（prompt_final 全文）：", _code(p.get("prompt_final"))]
        lines += [""]
    return lines


def _sec_audio(ctx: _Ctx) -> list:
    lines = ["", "## 六、音频链路", ""]
    ra = ctx.ref_audio
    if ra:
        lines += ["### 参考片音轨（copy_bgm / copy_reference_audio 管的是这条）", "",
                  "- 判定：有BGM=%s，有口播=%s → 策略「%s」（%s）"
                  % (ra.get("有背景音乐"), ra.get("有人声口播"), ra.get("策略"), ra.get("原因")),
                  "- 备好的 BGM 文件：`%s`" % (_disp(ctx.tdir, ra.get("bgm文件")) or "无"), ""]
    v = ctx.voice
    if v:
        lines += ["### 音色基准（克隆配音与 AI 段参考音都用它，与参考片原声无关）", "",
                  "- 策略：%s（来源：%s）" % (v.get("策略"), v.get("来源") or "无"),
                  "- 基准来源：片段 `%s`，%s" % (v.get("基准来源片段") or "无",
                                                 v.get("基准来源说明") or ""),
                  "- 处理：%s" % (v.get("基准音处理") or "未处理"),
                  "- 文件：`%s`（响度 %s dB）" % (_disp(ctx.tdir, v.get("基准音文件")) or "无",
                                                  v.get("基准音响度dB"))]
        if v.get("候选弃用"):
            lines += ["- 弃用候选：" + "；".join("%s（%s）" % (r.get("片段"), r.get("原因"))
                                                for r in v["候选弃用"])]
        lines += [""]
    return lines


def _sec_final(ctx: _Ctx) -> list:
    res = ctx.rec.get("result") or {}
    if not res:
        return ["", "## 七、成片", "", "任务尚未走到合成步骤，暂无成片。"]
    subs = res.get("subtitles") or {}
    return ["", "## 七、成片", "",
            "- 成片：`%s`（%s）" % (res.get("final"), _sec_fmt(res.get("duration_sec"))),
            "- 用段：%s/%s" % (res.get("segments_used"), res.get("segments_planned")),
            "- 音轨：%s" % json.dumps(res.get("audio") or {}, ensure_ascii=False)[:300],
            "- 字幕：%s%s" % (subs.get("mode") or "未处理",
                              "（已烧入）" if subs.get("burned") else "")]


def build(task_id: str) -> str:
    """生成 report.md，返回绝对路径。产物缺什么章节就少什么，不抛异常打断流水线。

    单章节容错本来就有，但最容易失败的两件事恰好在保护范围外：读任务目录（_Ctx 里每个
    产物 json 的顶层类型都可能与预期不符）和写 report.md（目录可能压根不存在）。
    这两处也兜住，才对得起「跑了一半失败的任务也能出报告」这句承诺。
    """
    try:
        ctx = _Ctx(task_id)
    except Exception as exc:  # noqa: BLE001
        tdir = _task_dir(task_id)
        os.makedirs(tdir, exist_ok=True)
        out = os.path.join(tdir, "report.md")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("# 任务报告 %s\n\n> 读取任务产物失败：%s\n" % (task_id, str(exc)[:300]))
        return out
    lines = []
    for sec in (_sec_head, _sec_inputs, _sec_product_images, _sec_script_audit, _sec_assets,
                _sec_match, _sec_segments, _sec_audio, _sec_final):
        try:
            lines += sec(ctx)
        except Exception as exc:  # noqa: BLE001  单章失败不拖垮整份报告
            lines += ["", "> （%s 章节生成失败：%s）" % (sec.__name__, str(exc)[:160])]
    os.makedirs(ctx.tdir, exist_ok=True)
    out = os.path.join(ctx.tdir, "report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    return out


def main() -> int:
    """命令行入口：python3 report.py [task_id]，给指定或最新任务出报告。"""
    if len(sys.argv) > 1:
        tid = sys.argv[1]
    else:
        tasks = sorted((p for root in TASK_ROOTS
                        for p in glob.glob(os.path.join(root, "*", "task.json"))),
                       key=os.path.getmtime, reverse=True)
        if not tasks:
            print("没有任务")
            return 1
        tid = os.path.basename(os.path.dirname(tasks[0]))
    print(build(tid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
