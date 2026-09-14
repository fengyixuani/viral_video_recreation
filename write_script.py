"""按爆款视频的拆解结果，为自己的商品仿写一版剧本，并产出人物/场景设定图与 ≤15s 分段计划。

跑法：python3 write_script.py（参考片、商品信息、开关都在 main() 里直接改）

字段名统一中文，与 analyze_reference 的拆解 schema 保持同一套命名习惯。

三步流水线：
1. 仿写剧本：读参考的 镜头骨架 / 结构 / 节奏 / 情绪曲线，让 LLM 按同一套
   分镜节拍与运镜编排换题材重写，并逐条说明是怎么复刻的（复刻说明）。
2. 设定图：人物设定图走 line_art.gen_line_art_frame（涉及人脸必须用线稿能力，否则 seedance
   会被真人风控拒）；场景设定图不含人脸，直接 aigc.gen_image。
3. 分段：seedance 单次最长 15s，用动态规划把分镜打包成段，切点强制落在镜头边界，并优先
   落在「硬切」这种镜头切换处，避免在叠化/运镜衔接/匹配剪辑的连续画面中间断开。

产物：output/script_writing/{商品名}/ 下的 script.json、script.md、assets/*.jpg
"""
import concurrent.futures as cf
import json
import os
import sys
from typing import Any, Optional

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import analyze_reference  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import line_art  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]

OUT_ROOT = os.path.join(config.OUTPUT_DIR, "script_writing")

MAX_SEG = 15.0   # seedance 单次上限
MIN_SEG = 4.0    # seedance 下限，短于此会被 clamp 到 4s

# 这些转场说明前后两镜画面是连续的，在此处断段会让拼接出现不和谐的跳变
SOFT_TRANSITIONS = ("叠化", "溶解", "运镜衔接", "匹配剪辑", "转场衔接", "无缝", "遮罩转场",
                    "划像", "闪白", "闪黑", "同镜")
HARD_TRANSITIONS = ("硬切", "切换", "直切", "跳切", "无")

WRITE_SYSTEM = ("你是短视频爆款剧本操盘手，擅长把一条已验证的爆款结构换题材复刻。"
                "严格保留参考片的节拍与镜头骨架，只替换题材、人物、场景和产品。输出 json。")

# 复刻强度：strict = 只替换人物/商品/场景，creative = 允许大模型改写剧本
MODE_RULES = {
    "strict": """
【复刻强度：严格复刻】
- 这是元素替换任务，不是创作任务：分镜数量必须与参考片完全相同，一镜对一镜，不得增删合并。
- 每镜的时长、景别、运镜、转场、叙事功能必须与所对应的参考镜一致。
- 剧情走向、冲突设计、反转位置、台词的句数与语气节奏都照参考片走。
- 只允许替换：人物身份与外观、商品/道具、场景与时代风格，以及台词里与商品相关的具体信息。
- 不要新增剧情、不要改钩子形式、不要重新设计结构，不要自行加长或压缩节奏。""",
    "creative": """
【复刻强度：创意仿写】
- 复刻的是传播机制，不是画面：保留钩子类型、结构段落、节奏快慢分布、情绪曲线走向和产品植入位置。
- 允许重写人物关系、剧情、台词和场景设定，分镜数量可在参考片 ±20% 内浮动。
- 每镜仍必须写明「对应参考镜」，并在复刻说明里讲清替换了什么、为什么效果等价。""",
}

WRITE_PROMPT = """下面给你一条爆款视频的完整拆解，以及我要推广的商品信息。
请仿照这条爆款的分镜编排、创意逻辑、节拍和运镜，为我的商品写一版新剧本。

硬性要求：
- 全片总时长必须对齐参考片的 __REFTOTAL__ 秒（各镜时长之和，允许 ±10% 以内的偏差）。
- 花字：只有参考片分镜的「特效」里出现字幕/花字/文案类元素时才写「花字」（14字内，
  承接参考镜文案的信息功能）；参考片画面干净无字的，所有镜的「花字」留空——
  成片字幕跟随参考片，参考片没有口播也没有花字，成片就不该有任何叠加文字。
- 分镜数量、每镜时长、景别、运镜、转场方式要与参考片一一对应，
  参考片的节奏快慢分布必须保留。
- 每个新分镜都要标明它对应参考片的第几镜（对应参考镜）。
- 题材/朝代/人物/场景可以换，但情绪曲线的走向和强度必须对齐。
- 商品必须像参考片那样作为「破局关键道具」融入剧情，不要变成硬广口播。
- 商品在画面里的可见状态必须忠于「我的商品」的图片描述：屏幕/表盘/界面显示什么、指示灯
  与开合形态，只能写图片描述里实际看得见的样子；图片里没有的显示内容不要指定，
  用「屏幕亮起」这类中性写法。卖点文案宣称的软件功能、壁纸角色、界面动效是文字信息，
  只能进台词与字幕，不许当画面元素写进「画面」和「视频提示词」——素材里看不见的东西
  写进画面，生成端只能凭空编一个假商品。
- 参考片里对原商品形态的描写（无Logo素面、屏幕在播什么、开合与表面形变）是原商品的
  事实，不许照搬到新商品上：对应参考镜时保留镜头功能与节奏，商品自身的样子（印刷标识、
  部件、屏幕内容）一律按「我的商品」图片描述重写。
- 贯穿多镜或前后呼应的视觉元素（道具、光效、屏幕显示内容）先在「视觉锚点」里定死形态
  与颜色，分镜写到它们时照抄锚点描述，前后镜不许漂移（不许这一镜绿色光效、下一镜同一
  元素变成蓝色）；相互衔接的元素（如道具光效点亮的屏幕）颜色要呼应。
- 主角商品必须从图片描述里实际存在的配色/款式中选定一款，写成第一条「视觉锚点」
  （名称就是商品名，形态与颜色里写清选定的配色），此后所有出现商品的分镜都用这一款：
  不许沿用参考片原产品的颜色，不许中途无缘由换色。确要展示多配色时，单独安排明确的
  配色展示镜，且只用图片里真实存在的配色。
- 台词要能在这一镜的时长里念完：每镜台词字数不超过「时长秒 × __RATE__」（约 __RATE__ 字/秒），
  一镜说不完就把话分到后面几镜去，不要把长句塞进短镜。没有台词的镜写空字符串。
__MODE_RULES__

输出 json，字段名原样使用下面的中文键名，结构如下：
{
 "剧本名": "",
 "一句话概要": "一句话讲清新剧本讲什么",
 "核心创意": "核心创意，以及它和参考片创意的对应关系",
 "开场钩子": "开头3秒钩子",
 "结构": [{"名称": "", "时间区间": "", "作用": ""}],
 "音乐": "音乐风格、情绪、卡点安排",
 "情绪曲线": [{"时间": "", "情绪": "", "强度": 1}],
 "节奏": "总时长、镜头数、平均镜头时长、快慢分布",
 "人物设定": [{"编号": "c1", "姓名": "", "角色": "主角/对手/配角",
   "外观": "性别年龄、发型发色、上衣下装鞋子的款式与颜色、体型、随身道具（只写外观，不写五官表情）",
   "性格": "性格与说话方式", "转变": "在片中的转变"}],
 "场景设定": [{"编号": "s1", "名称": "", "描述": "地点、时代风格、关键道具陈设、天气时间",
   "光线": "光线方向与性质", "色调": "主色调"}],
 "视觉锚点": [{"名称": "贯穿多镜的道具/光效/屏幕显示内容", "形态与颜色": "固定描述，分镜必须照抄",
   "出现镜头": [2, 3]}],
 "分镜": [{"序号": 1, "对应参考镜": 1, "时长秒": 3.0, "场景编号": "s1", "人物编号": ["c1"],
   "景别": "景别", "运镜": "机位与运镜，含运动方向速度",
   "画面": "这一镜画面整体是什么样子，构图光线色调",
   "动作": "人物动作的起点与终点", "台词": "台词原文，无则空字符串",
   "花字": "这一镜叠加在画面上的文案（14字内），有台词的镜留空",
   "特效": ["特效/字幕花字"], "转场": "从上一镜进入这一镜的转场方式，第1镜写「无」",
   "音效音乐": "音效与音乐变化", "叙事功能": "叙事功能",
   "需要模特出镜": true,
   "视频提示词": "给视频生成模型的中文提示词，一段话写清运镜+画面+人物动作+光线情绪，不要写时长和画幅"}],
 "复刻说明": [{"维度": "复刻维度，如 镜头骨架/节拍/情绪曲线/钩子/产品植入/运镜",
   "参考片": "参考片是怎么做的", "本片": "我们怎么做的", "原理": "为什么这样能复刻住效果"}]
}

【参考片拆解】
__REF__

【我的商品】
__PRODUCT__
__PERSONS__
每个分镜的「需要模特出镜」必须是布尔值 true / false：这一镜的画面里必须出现真人模特的
人脸或上半身才能表达清楚 → true；只拍商品、手部、环境、文字、特写等不需要人脸 → false。
实在判不出来才写 null，不要写字符串。

严格只输出 json。"""

PRODUCT_IMG_PROMPT = ("描述这张商品图里的商品：品类、外形结构、材质质感、主色与配色、机身或包装上印刷的显著文字、"
                      "屏幕/表盘当前实际显示的内容、大致尺寸感。只写画面里看得见的商品本身，不写背景和光线。"
                      "图上叠加的营销文案属于文字宣传：它宣称的功能画面（壁纸角色、界面动效等）画面里没有"
                      "就不要写，更不要当成屏幕显示内容。不超过100字，直接输出描述。")

SCRIPT_AUDIT_PROMPT = """审查一版新剧本。它是照一条爆款参考片的骨架、为「我的商品」仿写的。

【商品事实】（图片描述 = 商品图里实际看得见的内容）
__PRODUCT__

【视觉锚点】（剧本自己定下的全片统一元素，第一条应是主角商品及其选定配色）
__ANCHORS__

【参考片分镜】（骨架来源，用来判断复刻是否错位）
__REF__

【分镜】（只列了会进画面的字段）
__SHOTS__

逐镜检查四类违规：
一、商品状态失实（素材里看不见却被写成画面）：
- 屏幕/表盘/界面显示了图片描述里不存在的具体内容（壁纸角色、界面、图案、成段文字）
- 指示灯、开合形态、部件、印刷图案与图片描述冲突或凭空新增
- 商品配色/款式是图片描述里不存在的，或主角商品在不同分镜之间配色漂移
  （与锚点选定的配色不一致，又不是明确的多配色展示镜）
- 把卖点文案宣称的功能画面（软件效果、动效角色）写成了画面元素
二、商品漂移：某一镜的主体商品变成了别的品类/品牌/型号。
三、逻辑硬伤：前后镜动作、位置、道具状态自相矛盾（已经拿起又再次拿起、
  手里的东西凭空消失、状态无来由反复横跳）。
四、复刻错位：与「对应参考镜」的叙事功能明显不等价，破坏了钩子或节奏。
创意特效（光效、粒子、转场）、人物表演、环境氛围、字幕花字，以及「屏幕亮起」这类中性写法
都不算违规；确定是问题才报，不要吹毛求疵。

「视觉锚点」自己也要按同一套口径查：锚点的「形态与颜色」是分镜照抄的模板、也会随每段下发给
生成模型，锚点里写了图片描述里不存在的配色、部件、显示内容，等于全片一起失实。
查到就出一条 字段="视觉锚点" 的违规，「锚点名称」写清是哪一条，「改写」给这条锚点
「形态与颜色」的完整改写文本。

输出 json：
{"违规": [{"序号": 3, "类型": "商品状态失实/商品漂移/逻辑硬伤/复刻错位",
  "字段": "画面 或 视频提示词 或 视觉锚点", "锚点名称": "字段=视觉锚点 时必填，否则留空",
  "问题": "30字内",
  "改写": "该字段的完整改写文本：最小改动修掉问题，其余保持原样"}]}
每条违规的「改写」都必须给出非空的完整改写文本。字段=画面/视频提示词 时「序号」必须是分镜序号
（同一镜两个字段都违规就各出一条）；字段=视觉锚点 时按「锚点名称」定位，「序号」可以写 0。
没有违规就输出 {"违规": []}。只输出 json。"""


def _audit_product_states(script: "dict[str, Any]", product: "dict[str, Any]",
                          ref: "Optional[dict[str, Any]]" = None) -> "dict[str, Any]":
    """剧本层门禁：商品状态失实/商品漂移/逻辑硬伤/复刻错位，就地改写并留痕。

    WRITE_PROMPT 里的忠实约束是给模型的指令，偶尔会被无视；状态图那层只保护参考图，
    视频提示词直接让生成端画不存在的东西时只有这里能拦，验一遍才算闭环。
    审查/改写失败不阻断流水线，但结果显式写进 script["商品状态审查"]，
    review_case 按「未修复>0 或 审查失败」对账成问题。"""
    shots = script.get("分镜") or []
    if not shots:
        return script
    brief = [{k: s.get(k) for k in ("序号", "对应参考镜", "画面", "视频提示词",
                                    "动作", "叙事功能") if s.get(k)}
             for s in shots]
    info = {k: product.get(k) for k in ("name", "category", "appearance",
                                        "image_descriptions") if product.get(k)}
    ref_brief = [{k: s.get(k) for k in ("序号", "画面", "叙事功能") if s.get(k)}
                 for s in (ref or {}).get("分镜") or []]
    try:
        raw = aigc.understand(
            SCRIPT_AUDIT_PROMPT
            .replace("__PRODUCT__", json.dumps(info, ensure_ascii=False, indent=1))
            .replace("__ANCHORS__", json.dumps(script.get("视觉锚点") or [],
                                               ensure_ascii=False, indent=1))
            .replace("__REF__", json.dumps(ref_brief, ensure_ascii=False, indent=1))
            .replace("__SHOTS__", json.dumps(brief, ensure_ascii=False, indent=1)),
            max_tokens=6144, json_mode=True)
        found = _parse_json(raw).get("违规") or []
    except Exception as exc:  # noqa: BLE001
        script["商品状态审查"] = {"状态": "审查失败：%s" % str(exc)[:200], "违规": []}
        return script
    by_no = {s.get("序号"): s for s in shots}
    # 锚点按名称定位：锚点没有序号，而它一错就是全片错（分镜照抄它、每段下发它）
    anchors = [a for a in (script.get("视觉锚点") or []) if isinstance(a, dict)]
    by_name = {str(a.get("名称") or ""): a for a in anchors}
    unfixed = []
    for v in found:
        try:                       # 模型偶尔把序号回成字符串
            no = int(v.get("序号"))
        except (TypeError, ValueError):
            no = v.get("序号")
        field, text = v.get("字段"), (v.get("改写") or "").strip()
        if field == "视觉锚点" and text:
            a = by_name.get(str(v.get("锚点名称") or ""))
            if a:
                v["原文"] = a.get("形态与颜色")
                a["形态与颜色"] = text
                continue
            unfixed.append(v)
            continue
        s = by_no.get(no)
        if s and field in ("画面", "视频提示词") and text:
            v["原文"] = s.get(field)
            s[field] = text
        else:
            unfixed.append(v)
    script["商品状态审查"] = {
        "状态": "通过" if not found
        else "违规 %d 处，改写 %d 处，未修复 %d 处" % (len(found), len(found) - len(unfixed),
                                                        len(unfixed)),
        "违规": found, "未修复": unfixed}
    return script


# 整片叙事连贯性审查。跟「商品状态审查」互补：那一层逐镜查失实与矛盾，这一层只问整体
# ——28 个镜子各讲各的、连起来看不出在干什么，逐镜查是全绿的（实测 17_苹果笔记本 复刻小米
# 手机那条：28 镜 / 71.1s / **0 镜有台词**（参考片无口播，台词被规则表全清），
# 商品状态审查「违规6处改写6处未修复0处」，成片却没人看得懂）。
# 所以这里必须先做「盲测」：不给概要，只让模型顺着画面看一遍。
NARRATIVE_AUDIT_PROMPT = """审查一版短视频剧本的**整体叙事**能不能被看懂。

第一步（盲测，最重要）：先只看【分镜】按序号顺序播放的画面与台词，不要看下面的【剧本意图】，
用一句话说出「你认为这条片子在讲什么」。说不出来就直说说不出来。

第二步：再看【剧本意图】，对比你盲测的那句话，判断这条剧本是否兑现了它的意图。

【剧本意图】
__INTENT__

【结构】（剧本自己划的段落，每段有时间区间和「作用」）
__STRUCTURE__

【分镜】（按序播放的全片，时长秒为该镜时长）
__SHOTS__

判定要点：
- 观众看不看得懂，只取决于画面与台词本身。不要用【剧本意图】去补全你在画面里看不到的信息。
- 「理解断点」指：看到这一镜时观众不知道发生了什么、或它与前面建立的情境接不上
  （道具/场景/人物无来由地换了、动作没有交代过、突然跳到一个不相干的画面）。
- 「孤立镜」指：把这一镜整段删掉，对理解整片没有任何影响。
- 「缺失交代」指：相邻两镜之间少了一个必要的过渡/交代镜，观众会跟不上。
- 快切、无台词、抽象视觉本身不是问题；只有**导致看不懂**才算问题。
- 只报确定的问题，不要为了凑数报。没有问题就把列表留空、可理解性给高分。

输出 json：
{"盲测概要": "只看画面台词得出的一句话；说不出就写「说不出这条片子在讲什么」",
 "可理解性": 0-100 的整数（100=不看意图也一眼看懂，60 以下=普通观众看不懂在干什么）,
 "意图兑现": "兑现 / 部分兑现 / 未兑现",
 "总体问题": "40字内说清最主要的一个问题；没问题写空字符串",
 "理解断点": [{"序号": 12, "问题": "30字内",
              "建议": "怎么改，可以是改画面/补台词或花字/加一个交代镜"}],
 "孤立镜": [{"序号": 6, "问题": "30字内"}],
 "缺失交代": [{"位置": "镜7与镜8之间", "缺了什么": "30字内", "建议": "30字内"}],
 "结构未兑现": [{"段名称": "悬念铺垫", "问题": "这一段覆盖的几镜没做到它声明的作用，30字内"}]}
只输出 json。"""

# 低于这个分就写进剧本告警：普通观众看不出在干什么，值得人工看一眼再决定要不要重跑
NARRATIVE_MIN_SCORE = 60

# 审查报出问题后的修补。只允许「改字」，不允许「改结构」——这是硬约束，代码层也会再卡一遍：
# 增删镜头会改变分镜数与总时长，而下游 match（按镜派素材）/generate（按段出片）/字幕（按镜摊
# 时间）全按镜工作，对应参考镜的映射也会整体错位。误判一次的代价远大于漏修一处。
NARRATIVE_REPAIR_PROMPT = """一版短视频剧本的整片叙事审查发现了问题，请做**最小改动**把它修顺。

【审查结论】
__VERDICT__

【分镜】（按序播放的全片）
__SHOTS__

【可改的字段】
- 画面：这一镜拍什么（给人看的描述）
- 视频提示词：下发给视频生成模型的那句话，改画面时必须同步改它，两者不能互相矛盾
- 台词：这一镜的念白__SPEECH_RULE__
- 花字：屏幕上叠的短文字__CAPTION_RULE__

【硬约束】（违反的条目会被直接丢弃）
- 不许增加、删除、合并、拆分、重排任何分镜。序号与时长秒一个都不能改。
- 只能改上面列出的字段，且只改真正有问题的那几镜，其余保持原样。
- 「缺失交代」不能靠加镜解决：改写相邻两镜的画面，把缺的那一环用动作交代进去
  （例如后一镜开头补上「手拿起 X」，让它与前一镜接得上）。
- 「孤立镜」不能删：把它改成与主线相关的画面（让它承接前一镜的动作或指向商品），
  时长不变。
- 改写要贴着原镜的景别、运镜、时长与节奏，不要把一个 1.5 秒的快切改成需要 5 秒才演完的动作。
- 画面里只能出现商品图里真实存在的东西，不许凭空新增屏幕显示内容、部件、配色。

输出 json：
{"修改": [{"序号": 6, "字段": "画面", "理由": "20字内说明修掉的是哪个问题",
           "改写": "该字段的完整改写文本"}],
 "说明": "40字内总述这次修补做了什么"}
没有可以在不动结构的前提下修掉的问题，就输出 {"修改": [], "说明": "原因"}。只输出 json。"""


def _apply_narrative_fix(shot: "dict[str, Any]", field: str, text: str,
                         allowed: "list[str]") -> str:
    """把一条修补落到分镜上。落不下去返回不采纳的原因，成功返回空串。

    代码层再卡一遍提示词里的硬约束：模型的自我约束不可靠，而这里改的是会直接下发给
    生成端的字段，宁可少修一处。
    """
    if field not in allowed:
        return "字段「%s」不在允许改写的范围内" % field
    if not text:
        return "改写文本为空"
    if field == "台词":
        room = _dur(shot) * SPEECH_RATE
        got = len([c for c in text.split("：", 1)[-1] if not c.isspace()])
        if got > room + 1:
            return "台词 %d 字超出这一镜 %.1fs 念得完的 %.0f 字" % (got, _dur(shot), room)
    if field == "花字" and len(text) > 14:
        return "花字 %d 字超出 14 字上限" % len(text)
    shot[field] = text
    return ""


def _narrative_repair(script: "dict[str, Any]", verdict: dict) -> dict:
    """按审查结论就地修补剧本，只改字段不动结构。返回修补记录。"""
    shots = script.get("分镜") or []
    by_no = {}
    for s in shots:
        try:
            by_no[int(str(s.get("序号")).strip())] = s
        except (TypeError, ValueError):
            continue
    allowed, speech_rule, caption_rule = _narrative_fields(script)
    brief = [{k: s.get(k) for k in ("序号", "时长秒", "景别", "运镜", "画面", "视频提示词",
                                    "台词", "花字") if s.get(k)}
             for s in shots]
    trace = {k: verdict.get(k) for k in ("总体问题", "理解断点", "孤立镜", "缺失交代",
                                         "结构未兑现") if verdict.get(k)}
    try:
        got = _understand_json(
            NARRATIVE_REPAIR_PROMPT
            .replace("__SPEECH_RULE__", speech_rule)
            .replace("__CAPTION_RULE__", caption_rule)
            .replace("__VERDICT__", json.dumps(trace, ensure_ascii=False, indent=1))
            .replace("__SHOTS__", json.dumps(brief, ensure_ascii=False, indent=1)),
            max_tokens=6144, json_mode=True, engine="qwen")
    except Exception as exc:  # noqa: BLE001
        return {"状态": "修补失败：%s" % str(exc)[:200], "已改": [], "未采纳": []}
    done, skipped = [], []
    for fix in (got.get("修改") or []):
        if not isinstance(fix, dict):
            continue
        try:
            no = int(str(fix.get("序号")).strip())
        except (TypeError, ValueError):
            no = None
        shot, field = by_no.get(no), str(fix.get("字段") or "").strip()
        text = str(fix.get("改写") or "").strip()
        if not shot:
            skipped.append(dict(fix, 未采纳原因="找不到序号 %s 的分镜" % fix.get("序号")))
            continue
        before = shot.get(field)
        why = _apply_narrative_fix(shot, field, text, allowed)
        if why:
            skipped.append(dict(fix, 未采纳原因=why))
        else:
            done.append({"序号": no, "字段": field, "理由": fix.get("理由") or "",
                         "原文": before, "改写": text})
    return {"状态": "已改 %d 处，未采纳 %d 处" % (len(done), len(skipped)),
            "说明": str(got.get("说明") or "").strip(),
            "可改字段": allowed, "已改": done, "未采纳": skipped}


def _narrative_fields(script: "dict[str, Any]") -> tuple:
    """这条剧本允许修补哪些字段，以及要写进提示词的两句口径说明。

    台词与花字不是随便能加的：
    - 参考片没有人声口播时「口播复刻」表已经把全片台词清空（rules 的口径），这里再加回去
      等于绕过规则表；
    - 花字同理，WRITE_PROMPT 的口径是「参考片画面有字才写」，凭空加字会破坏复刻一致性。
    所以这两个字段只在剧本本来就有它们的时候才开放，其余情况只允许改画面与视频提示词。
    """
    shots = script.get("分镜") or []
    voice_off = (script.get("口播判定") or {}).get("动作") == "不写台词"
    has_speech = any(str(s.get("台词") or "").strip() for s in shots)
    has_caption = any(str(s.get("花字") or "").strip() for s in shots)
    speech_ok = has_speech and not voice_off
    fields = ["画面", "视频提示词"] + (["台词"] if speech_ok else []) + \
             (["花字"] if has_caption else [])
    speech_rule = ("（可改，但字数不能超过 时长秒 × %.1f，否则念不完）" % SPEECH_RATE
                   if speech_ok else "（本片按复刻口径不写台词，**不许**给任何镜加台词）")
    caption_rule = ("（可改，14 字内）" if has_caption
                    else "（参考片画面本来没有文字，**不许**给任何镜加花字）")
    return fields, speech_rule, caption_rule


def _narrative_check(script: "dict[str, Any]") -> dict:
    """跑一次整片叙事连贯性审查，返回归一化后的结论（不写回 script）。

    走纯文本链路（qwen）：输入全是文字，不需要视觉理解，没必要占 gemini 的视频额度。
    """
    shots = script.get("分镜") or []
    intent = {k: script.get(k) for k in ("剧本名", "一句话概要", "核心创意", "开场钩子",
                                         "情绪曲线", "节奏") if script.get(k)}
    # 台词必须带上：它是「看懂」的主要载体之一，而商品状态审查那层是不看台词的
    brief = [{k: s.get(k) for k in ("序号", "时长秒", "画面", "台词", "花字",
                                    "叙事功能") if s.get(k)}
             for s in shots]
    try:
        got = _understand_json(
            NARRATIVE_AUDIT_PROMPT
            .replace("__INTENT__", json.dumps(intent, ensure_ascii=False, indent=1))
            .replace("__STRUCTURE__", json.dumps(script.get("结构") or [],
                                                 ensure_ascii=False, indent=1))
            .replace("__SHOTS__", json.dumps(brief, ensure_ascii=False, indent=1)),
            max_tokens=4096, json_mode=True, engine="qwen")
    except Exception as exc:  # noqa: BLE001
        return {"状态": "审查失败：%s" % str(exc)[:200]}
    try:
        score = int(float(got.get("可理解性")))
    except (TypeError, ValueError):
        score = -1                      # 模型没给分：记下来但不当成低分误报告警
    out = {"状态": "通过", "可理解性": score,
           "盲测概要": str(got.get("盲测概要") or "").strip(),
           "意图兑现": str(got.get("意图兑现") or "").strip(),
           "总体问题": str(got.get("总体问题") or "").strip()}
    for key in ("理解断点", "孤立镜", "缺失交代", "结构未兑现"):
        items = [x for x in (got.get(key) or []) if isinstance(x, dict)]
        if items:
            out[key] = items
    hits = sum(len(out.get(k) or []) for k in ("理解断点", "孤立镜", "缺失交代", "结构未兑现"))
    out["问题数"] = hits
    if 0 <= score < NARRATIVE_MIN_SCORE or out["意图兑现"] == "未兑现":
        out["状态"] = "可理解性不足"
    elif hits:
        out["状态"] = "有 %d 处可优化" % hits
    return out


def _audit_narrative(script: "dict[str, Any]") -> "dict[str, Any]":
    """整片叙事连贯性：审查 → 就地修补 → 复审，结果写 script["叙事连贯审查"]。

    修补只改字段（画面/视频提示词，以及本来就有的台词/花字），绝不增删镜头、不改序号与
    时长——下游 match 按镜派素材、generate 按段出片、字幕按镜摊时间，动结构会整体错位，
    「对应参考镜」的映射也会失效。所以「缺失交代」靠改写相邻两镜的画面把那一环演出来，
    「孤立镜」靠改写让它接回主线，而不是加镜删镜。约束在提示词里写明，代码层再卡一遍
    （见 _apply_narrative_fix），卡不过的进「未采纳」。

    复审后仍然不达标只写剧本告警、不阻断流水线：一次模型误判不该让整条任务失败，
    这类问题人看一眼就能确认（review_case 对账、report.md 展示、日志打印）。
    """
    shots = script.get("分镜") or []
    if not shots:
        return script
    first = _narrative_check(script)
    out = dict(first)
    state = str(first.get("状态") or "")
    if state != "通过" and not state.startswith("审查失败"):
        fix = _narrative_repair(script, first)
        out["修补"] = fix
        if fix.get("已改"):
            again = _narrative_check(script)
            out["复审"] = again
            # 复审分更高才认修补结果；退步就照旧按首轮结论报，避免"修完更差还显示变好"
            if not str(again.get("状态") or "").startswith("审查失败"):
                if again.get("可理解性", -1) >= first.get("可理解性", -1):
                    out.update({k: again.get(k) for k in
                                ("状态", "可理解性", "盲测概要", "意图兑现", "总体问题",
                                 "问题数")})
                    for key in ("理解断点", "孤立镜", "缺失交代", "结构未兑现"):
                        out.pop(key, None)
                        if again.get(key):
                            out[key] = again[key]
                else:
                    out["复审说明"] = "复审分数没有提高，最终结论按首轮报"
    out["首轮"] = {k: v for k, v in first.items() if k != "首轮"}
    if out.get("状态") == "可理解性不足":
        script.setdefault("剧本告警", []).append(
            "整片叙事连贯性可理解性 %s 分（低于 %d）%s：%s；盲测看成「%s」"
            % (out.get("可理解性") if out.get("可理解性", -1) >= 0 else "未给出",
               NARRATIVE_MIN_SCORE,
               "，已自动修补 %d 处仍不达标" % len(out.get("修补", {}).get("已改") or [])
               if (out.get("修补") or {}).get("已改") else "",
               out.get("总体问题") or "见叙事连贯审查", out.get("盲测概要") or "-"))
    script["叙事连贯审查"] = out
    return script


# 设定图里的商品没有任何验收就会被当锚点下发（人物图排在 @图片1，权重最高），一台印错字、
# 混了配色的商品会被视频模型原样照抄。白底图与中间状态图都过「商品一致」比对，设定图必须补上。
ASSET_PRODUCT_CHECK_PROMPT = """第一张是商品原图，第二张是新生成的设定图（人物设定图或场景概念图），
里面应该出现同一个商品。判断第二张里的商品与原图是不是同一个商品，输出 json：

{"一致": true/false,
 "问题": "外形结构、部件位置、配色、机身或包装印刷文字（被改写/镜像反写/画成乱码）、
 有没有混进原图里没有的其它配色或款式；都没问题就写空字符串"}

只看商品本身：人物、背景、光线、构图、拍摄角度、商品在画面里的大小与位置都不算问题。
印刷文字被镜像、缺字、画成近似字符，以及同一个画面里出现多种配色或多个款式，一律判 一致=false。
第二张里根本没出现这个商品也判 一致=false，问题里写「设定图里没有商品」。只输出 json。"""

ASSET_AVOID_HINT = ("特别注意：上一轮生成的商品出现了这些走样，必须避免：%s。"
                    "商品的造型、部件位置、配色与机身印刷文字严格照参考图，文字不许镜像或改写。")


def _check_asset_product(ref_url: str, img_url: str) -> "dict[str, Any]":
    """设定图里的商品与主锚点原图比一遍。

    返回 {"一致": True/False/None, "问题": str}；None = 核验这步本身失败，按通过处理——
    不能因为一次调用失败就把设定图链路清空，那比一张可能走样的图更糟。
    """
    try:
        got = _parse_json(aigc.understand(
            ASSET_PRODUCT_CHECK_PROMPT,
            media=[{"type": "image", "url": ref_url}, {"type": "image", "url": img_url}],
            max_tokens=512, json_mode=True))
    except Exception as exc:  # noqa: BLE001
        return {"一致": None, "问题": "核验失败：%s" % str(exc)[:120]}
    return {"一致": rules.as_bool(got.get("一致")), "问题": str(got.get("问题") or "")}


# 用户上传人物参考图时，剧本里的人物必须按参考图的人来写，替换参考片原主角。
PERSON_REF_BLOCK = """
【人物参考】（用户指定的出镜人物，按角色戏份从主到次依次分配）
__LIST__
- 主角（以及依次往后的角色）的人物设定必须就是对应人物参考里的这个人：
  「外观」按参考描述写性别年龄、发型发色、体型与气质，不得沿用参考片原主角的长相特征；
  服装鞋履可按剧情与场景需要重新设计。
- 分镜「视频提示词」中的人物外观必须与该人物设定一致。
"""


def _parse_json(raw: str) -> "dict[str, Any]":
    txt = raw.strip().strip("`")
    if txt.startswith("json"):
        txt = txt[4:]
    return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])


def _understand_json(user: str, retries: int = 2, **kwargs) -> "dict[str, Any]":
    """问 LLM 要 json 并解析；格式坏了带着报错重问，最多 retries 次。

    模型输出偶发缺冒号/尾逗号这类语法抖动，重问一次基本都能好。全部失败才抛，
    让步骤显式失败而不是静默吞掉。"""
    ask = user
    for attempt in range(retries + 1):
        raw = aigc.understand(ask, **kwargs)
        try:
            return _parse_json(raw)
        except ValueError as exc:
            if attempt >= retries:
                raise
            print("json 解析失败（%s），重问一次" % str(exc)[:80], flush=True)
            ask = (user + "\n\n注意：上一次输出的 json 有语法错误（%s），"
                   "这次必须输出完整、合法、可被 json.loads 解析的 json。" % str(exc)[:120])


# ---------------- 第 1 步：仿写剧本 ----------------
def describe_products(images: "Optional[list[str]]") -> "list[dict[str, Any]]":
    """把商品图读成文字描述，供 LLM 写剧本时准确描述道具外观。"""
    def one(p: str) -> "dict[str, Any]":
        url = storage.upload(p) if os.path.isfile(p) else p
        return {"image": p, "url": url,
                "desc": aigc.understand(PRODUCT_IMG_PROMPT,
                                        media=[{"type": "image", "url": url}]).strip()}

    if not images:
        return []
    with cf.ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, images))


def describe_persons(images: "Optional[list[str]]") -> "list[dict[str, Any]]":
    """把人物参考图读成外观描述（发型/服装/体型/配色，不含五官），
    剧本人物设定与设定图据此对齐到同一个人。"""
    def one(p: str) -> "dict[str, Any]":
        url = storage.upload(p) if os.path.isfile(p) else p
        return {"image": p, "url": url, "desc": line_art.describe_person(url)}

    if not images:
        return []
    with cf.ThreadPoolExecutor(2) as ex:
        return list(ex.map(one, images))


def _persons_block(persons: "Optional[list[dict[str, Any]]]") -> str:
    if not persons:
        return ""
    lines = "\n".join("人物参考%d：%s" % (i, p.get("desc") or "")
                      for i, p in enumerate(persons, 1))
    return PERSON_REF_BLOCK.replace("__LIST__", lines)


def _ref_brief(ref: "dict[str, Any]") -> str:
    """喂给 LLM 的参考片摘要：整体分析全给，分镜只给编排相关字段，省 token 也避免它照抄画面。

    拆解产物用中文键名（analyze_reference 的 schema），老产物的英文键名由 normalize() 统一映射。
    """
    ref = analyze_reference.normalize(ref)
    keep = ("序号", "时长秒", "景别", "运镜", "转场", "叙事功能", "画面", "动作", "台词",
            "特效")   # 特效里有没有字幕/花字决定新剧本要不要写「花字」
    brief = {"整体": ref["整体"],
             "分镜": [{k: s[k] for k in keep if s.get(k)} for s in ref["分镜"]]}
    return json.dumps(brief, ensure_ascii=False, indent=1)


def _ref_total(ref: "dict[str, Any]") -> float:
    """参考片总时长：优先用拆解记录的真实时长，退化到各镜时长求和。"""
    total = float(ref.get("总时长秒") or 0)
    if total > 0:
        return total
    return sum(_dur(s) for s in analyze_reference.normalize(ref)["分镜"])


def _match_total(script: "dict[str, Any]", ref_total: float) -> "dict[str, Any]":
    """把剧本各镜时长等比缩放到参考片总时长，保证成片长度和参考片差不多。

    LLM 只会照抄参考片里每镜的秒数，一旦它自己又压短一轮，成片就明显短于参考片，
    所以这里做一次确定性校正，节奏比例保持不变。
    """
    shots = script.get("分镜") or []
    total = sum(_dur(s) for s in shots)
    if ref_total > 0 and total > 0 and abs(total - ref_total) / ref_total > 0.1:
        k = ref_total / total
        for s in shots:
            s["时长秒"] = round(max(0.3, _dur(s) * k), 1)
    return script


# 口播语速上限（字/秒）：本地克隆音色实测约 5.2 字/秒，配 1.25 倍加速仍自然，取 6.5。
# 参考片主播语速再快也不能照抄——我们的成片是拿这个语速念出来的。
SPEECH_RATE = 6.5


def _speech_chars(shot: "dict[str, Any]") -> int:
    """这一镜要念多少字（去掉「角色：」前缀和空白，标点也算，念的时候要停顿）。"""
    text = str(shot.get("台词") or "").strip()
    if "：" in text[:6]:
        text = text.split("：", 1)[-1]
    return len([c for c in text if not c.isspace()])


def _fit_speech(script: "dict[str, Any]") -> "dict[str, Any]":
    """把镜头时长按台词字数重分配，让每镜的话都念得完，总时长不变。

    LLM 常把长句塞进短镜（实测出现 25 字 /1.8s = 14 字/秒），出片时只能靠加速念白和放长
    画面硬塞，塞不下就把半句话切掉。所以这里先在剧本层把时间从「话少的镜」挪给「话多的镜」：
    总时长守恒 → 成片长度仍对齐参考片，只是内部节奏按念白需要微调。
    """
    shots = script.get("分镜") or []
    # 按下标做键，不用「序号」：序号是模型产出的，可能缺（s["序号"] 硬取键直接 KeyError）、
    # 也可能重复（两镜塌缩成一条，时长被分配到错误的镜上而不报错）。
    need = {}
    for i, s in enumerate(shots):
        chars = _speech_chars(s)
        if chars:
            need[i] = chars / SPEECH_RATE
    short = [i for i, s in enumerate(shots) if _dur(s) < need.get(i, 0) - 0.05]
    if not short:
        return script
    tight = set(short)
    deficit = sum(need[i] - _dur(shots[i]) for i in short)
    # 可让出的时间：话少/无台词的镜，各自最多让出一半，且不低于 0.4s（再短画面就没法看）
    donors = [(i, max(0.0, min(_dur(shots[i]) - need.get(i, 0), _dur(shots[i]) * 0.5,
                               _dur(shots[i]) - 0.4)))
              for i in range(len(shots)) if i not in tight]
    slack = sum(x[1] for x in donors)
    take = min(deficit, slack)
    if take > 0.05:
        for i, room in donors:
            if room > 0:
                shots[i]["时长秒"] = round(_dur(shots[i]) - room * take / slack, 1)
        for i in short:
            gain = (need[i] - _dur(shots[i])) * take / deficit
            shots[i]["时长秒"] = round(_dur(shots[i]) + gain, 1)
    left = [(shots[i].get("序号"), round(need[i] - _dur(shots[i]), 1)) for i in short
            if _dur(shots[i]) < need[i] - 0.3]
    if left:
        script.setdefault("剧本告警", []).append(
            "这些镜的台词仍超出时长（镜号, 还差秒）：%s；出片时会靠加速念白补" % left)
    return script


SHOT_MAX = 6.0   # 单镜上限：匹配/裁剪/字幕都是按镜工作的，太长的镜没有素材能对上（见 _cut_long_shots）


def _shot_no(shot) -> int:
    """分镜序号收敛成 int，专给排序用。

    序号是模型产出的字段：偶尔回成字符串（全字符串时按字典序排，"10" 会插到 "2" 前面，
    整条剧本顺序错乱且没有任何告警；int 与 str 混着回还会让 sorted 抛 TypeError，
    整个剧本步骤失败）。缺号的排到最后，不打断排序。
    """
    try:
        return int(str((shot or {}).get("序号")).strip())
    except (TypeError, ValueError):
        return 1 << 30


def _sentences(text: str, with_comma: bool = False) -> "list[str]":
    """按句读切句，标点跟着前一句走（念白要按句分给子镜，不能切在词中间）。

    with_comma=True 时逗号也算切点：一整镜只有一个长句时，只按句号切不出足够的份数。
    """
    stops = "。！？；…!?;" + ("，,、" if with_comma else "")
    out, cur = [], ""
    for ch in str(text or ""):
        cur += ch
        if ch in stops:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def _deal_sentences(lines: "list[str]", n: int) -> "list[str]":
    """把几句话按字数均匀切成最多 n 份，顺序不变，句子不拆开。

    每收一组就按「剩余字数 / 剩余组数」重算下一组的目标，并保证剩下的句子够给剩下的组各一句
    ——固定目标线的写法会让最后一组吃掉一大半（实测 152 字切 4 份变成 50/34/68）。
    """
    lines = [x for x in lines if x]
    if not lines or n <= 1:
        return ["".join(lines)] or [""]
    n = min(n, len(lines))
    groups, cur, rest = [], "", sum(len(x) for x in lines)
    for k, line in enumerate(lines):
        cur += line
        need = n - len(groups)
        if need > 1 and (len(lines) - k <= need or len(cur) >= rest / need):
            groups.append(cur)
            rest -= len(cur)
            cur = ""
    if cur:
        groups.append(cur)
    return groups



def _share_durs(total: float, sizes: "list[int]", cap: float = SHOT_MAX) -> "list[float]":
    """按字数比例把 total 秒分给各子镜，每份夹在 [0.4, cap] 之间，合计仍然等于 total。

    上限不能省：cap（SHOT_MAX）本来就是切镜的全部目的——下游素材匹配/裁剪/字幕都按镜工作。
    只按字数比例分的话「一个长句 + 好。」这种台词会分出 7.5s 的子镜（8s 镜切两份），
    「最后一份吃掉余数」更能分出 20s 的子镜（25s 镜切五份、前四份都是短句），
    等于切了个假的，还会跟着写下「已切成连续子镜」的告警。
    份数实在不够（cap * 份数 < total）时按均摊超出，至少不会集中砸在某一份上。
    """
    n = len(sizes)
    if n <= 0:
        return []
    lo = min(0.4, total / n)              # total 小到连 0.4 都摊不匀时以合计为准
    hi = max(cap, total / n)              # 份数不够时只能超上限，均摊到最小超出
    base = sum(sizes)
    durs = [(total * s / base if base else total / n) for s in sizes]
    # 夹进区间 → 把夹出来的差额摊给还有余量的子镜 → 再夹，直到合计收敛回 total
    for _ in range(n + 2):
        durs = [min(hi, max(lo, d)) for d in durs]
        gap = total - sum(durs)
        if abs(gap) < 0.01:
            break
        room = [(hi - d) if gap > 0 else (d - lo) for d in durs]
        avail = sum(room)
        if avail < 0.01:
            break
        durs = [d + gap * r / avail for d, r in zip(durs, room)]
    durs = [round(d, 1) for d in durs]
    drift = round(total - sum(durs), 1)   # 各份单独取整后的零头，补给最有余量的那一份
    if abs(drift) >= 0.05:
        i = max(range(n), key=lambda k: (hi - durs[k]) if drift > 0 else durs[k])
        durs[i] = round(durs[i] + drift, 1)
    return durs


def _cut_long_shots(script: "dict[str, Any]") -> "dict[str, Any]":
    """把超过 SHOT_MAX 的镜头切成连续子镜，序号重排，总时长不变。

    参考片里的「一镜到底口播」会被拆解成一个 20 多秒的镜；下游全部按镜工作——
    素材匹配拿一整镜去找素材（用户素材都是几秒的片段，必然一条都对不上，实测利用率 0）、
    裁剪按镜裁、字幕按镜摊时间。所以在剧本层先切开：画面/提示词照抄（本来就是同一镜的续拍），
    台词按句分给子镜，时长按分到的字数比例给（没台词就均分，见 _share_durs）。
    """
    out, changed = [], 0
    for shot in script.get("分镜") or []:
        total = _dur(shot)
        if total <= SHOT_MAX:
            out.append(shot)
            continue
        n = int(total // SHOT_MAX) + (1 if total % SHOT_MAX > 0.05 else 0)
        if n <= 1:
            # 6.00~6.05s 这一档：算下来只有 1 份，等于什么都没切。原来照旧往下走，
            # 会给这一镜盖上「同镜续拍 1/1」、把全片序号重排一遍，还写一条「已切成
            # 连续子镜」的告警——报了个没发生的事。
            out.append(shot)
            continue
        text = str(shot.get("台词") or "").strip()
        if not text:
            parts = [""] * n                      # 没台词的长镜（纯画面）按时长均分
        else:
            parts = _deal_sentences(_sentences(text), n)
            if len(parts) < n:                    # 句号切不够份数，就连逗号一起切
                parts = _deal_sentences(_sentences(text, with_comma=True), n)
            # 连逗号也切不出 n 份：补几个纯画面的续拍子镜，让每份都能压在 SHOT_MAX 以内
            parts += [""] * (n - len(parts))
        durs = _share_durs(total, [len(p) for p in parts])
        for i, text in enumerate(parts):
            sub = dict(shot)
            sub.update({"时长秒": durs[i], "台词": text,
                        "同镜续拍": "%d/%d" % (i + 1, len(parts))})
            if i:
                sub["转场"] = "同镜续拍"
            out.append(sub)
        changed += 1
    if changed:
        for i, shot in enumerate(out, 1):
            shot["序号"] = i
        script["分镜"] = out
        script.setdefault("剧本告警", []).append(
            "有 %d 个镜头超过 %.0fs，已切成连续子镜（共 %d 镜）以便匹配用户素材"
            % (changed, SHOT_MAX, len(out)))
    return script


def write_script(ref: "dict[str, Any]", product: "dict[str, Any]",
                 mode: str = "creative",
                 persons: "Optional[list[dict[str, Any]]]" = None) -> "dict[str, Any]":
    """让 LLM 按参考片的骨架为商品仿写剧本，总时长对齐参考片。

    mode="strict" 只替换人物/商品/场景，"creative" 允许改写剧本。
    persons 为 describe_persons() 的产物：给了就把主角（依次往后）钉死成参考人物。
    """
    info = dict(product)
    info.pop("images", None)
    ref_total = _ref_total(ref)
    # 口播跟随参考片：参考片纯 BGM/音效的，一句台词都不写，后面的配音、字幕自动跟着没有
    voice = rules.decide("口播复刻", {"参考片有人声口播": rules.as_bool(
        (analyze_reference.normalize(ref).get("整体") or {}).get("有人声口播"))})
    prompt = (WRITE_PROMPT.replace("__REF__", _ref_brief(ref))
                          .replace("__MODE_RULES__", MODE_RULES.get(mode, MODE_RULES["creative"]))
                          .replace("__REFTOTAL__", "%.1f" % ref_total)
                          .replace("__RATE__", "%.1f" % SPEECH_RATE)
                          .replace("__PRODUCT__", json.dumps(info, ensure_ascii=False, indent=1))
                          .replace("__PERSONS__", _persons_block(persons)))
    if voice["动作"] == "不写台词":
        prompt += ("\n\n【口播判定】参考片没有人声口播，本片同样不要口播：所有镜的「台词」"
                   "必须是空字符串，信息全靠画面、动作与花字（若参考片有花字）传达。")
    script = _understand_json(prompt, system=WRITE_SYSTEM, max_tokens=16384, json_mode=True)
    script["口播判定"] = voice                     # 判定痕迹落进剧本，报告/排查可对账
    if voice["动作"] == "不写台词":                 # 代码层兜底：模型硬写的台词一律清空
        for shot in script.get("分镜") or []:
            shot["台词"] = ""
    script["分镜"] = sorted(script.get("分镜") or [], key=_shot_no)
    for shot in script["分镜"]:
        # rules.py 的「切片缺失补片」表要读布尔，判不出来留 None 让通配吃掉（见 rules.decide）
        shot["需要模特出镜"] = rules.as_bool(shot.get("需要模特出镜"))
    # 先把总时长钉回参考片，再切开过长的镜，最后按台词字数在片内重分配（重分配总时长守恒）
    script = _fit_speech(_cut_long_shots(_match_total(script, ref_total)))
    # 分镜定稿后过两层剧本门禁，都只留痕不阻断：
    # 1) 逐镜：状态失实/漂移/逻辑硬伤/复刻错位，就地改写
    # 2) 整片：连起来能不能看懂（逐镜全绿但整片各讲各的，只有这一层查得出）
    #    放在改写之后，审的是最终会下发给生成端的那一版分镜
    return _audit_narrative(_audit_product_states(script, info, ref))


# ---------------- 第 2 步：人物 / 场景设定图 ----------------
# 设定图不能凭空编商品：文生图画出来的假商品会被 seedance 当参考照抄进成片。
# 所以先判断每张设定图里会不会出现广告商品，会出现的就把商品原图当参考图生图，
# 提示词里在商品出现的位置标 @图N 锚点（锚点编号 = ref_images 顺序）。
ASSET_REF_PROMPT = """下面是一版短视频剧本的人物设定、场景设定与分镜，以及要植入的商品。
逐个判断：这张设定图的画面里会不会出现这个商品本身（含它的包装）。

判断口径：
- 人物随身携带、手持、穿戴、正在使用这个商品 → 出现（设定图要把商品一起画进去）
- 场景的关键道具陈设里有这个商品 → 出现
- 设定描述里没写，但分镜里这个人物/场景多次和商品同框 → 出现
- 只是提到品类的同类物品、但明确不是这个商品 → 不出现
- 分镜里也从不和商品同框 → 不出现

出现的，改写它的生图提示词：在商品出现的位置紧跟 @图1 锚点（如「手中握着 @图1 的饮料罐」），
其余描述保持原意，不要新增画面元素。不出现的，提示词写空字符串。

输出 json：
{"资产": [{"编号": "c1", "出现商品": true, "提示词": "..."}]}

【商品】
__PRODUCT__

【人物设定】
__CHARS__

【场景设定】
__SCENES__

【分镜（判断同框用）】
__SHOTS__

只输出 json。"""


def plan_asset_refs(script: "dict[str, Any]", product: "dict[str, Any]") -> "dict[str, Any]":
    """批量判断每张设定图里是否出现商品，出现的给出带 @图N 锚点的提示词。

    一次调用问完所有人物与场景。失败时退回商品名/品类关键词匹配（只判断、不改写提示词），
    再失败就当作不出现商品，走原来的纯文生图，不阻塞出图。
    """
    chars = script.get("人物设定") or []
    scenes = script.get("场景设定") or []
    keys = ("name", "category", "appearance", "usage_scene")
    brief = {k: product.get(k) for k in keys if product.get(k)}
    shots = [{k: s.get(k) for k in ("序号", "场景编号", "人物编号", "画面", "动作") if s.get(k)}
             for s in (script.get("分镜") or [])]
    plan: "dict[str, Any]" = {}
    try:
        raw = aigc.understand(
            ASSET_REF_PROMPT
            .replace("__PRODUCT__", json.dumps(brief, ensure_ascii=False, indent=1))
            .replace("__CHARS__", json.dumps(chars, ensure_ascii=False, indent=1))
            .replace("__SCENES__", json.dumps(scenes, ensure_ascii=False, indent=1))
            .replace("__SHOTS__", json.dumps(shots, ensure_ascii=False, indent=1)),
            max_tokens=4096, json_mode=True)
        for item in _parse_json(raw).get("资产") or []:
            plan[str(item.get("编号"))] = {"出现商品": bool(item.get("出现商品")),
                                           "提示词": (item.get("提示词") or "").strip(),
                                           "判定来源": "LLM"}
    except Exception as exc:  # noqa: BLE001
        # 兜底只做判断不改写：商品名/品类关键词，在这张图自己的描述 + 引用到它的分镜里找。
        name = (product.get("name") or "").strip()
        words = {w for w in [name, name[-2:], name[-3:]] if len(w) >= 2}
        words |= {w.strip() for w in (product.get("category") or "").replace("/", " ").split()
                  if len(w.strip()) >= 2}
        for it, field, key in ([(c, "外观", "人物编号") for c in chars]
                               + [(s, "描述", "场景编号") for s in scenes]):
            cid = str(it.get("编号"))
            text = " ".join([str(it.get(field) or ""), str(it.get("角色") or "")]
                            + [str(s.get("画面") or "") + str(s.get("动作") or "")
                               for s in (script.get("分镜") or [])
                               if cid in (s.get(key) or [])])
            plan[cid] = {"出现商品": any(w in text for w in words),
                         "提示词": "", "判定来源": "关键词兜底：%s" % str(exc)[:120]}
    return plan


# 参考图生图必须写清「从参考图保留什么」，只给锚点模型会自己发挥。见 Skills/ImagePromptSkill.md
def _ref_declare(n: int) -> str:
    imgs = "、".join("@图%d" % (i + 1) for i in range(n))
    return "%s 是商品原图，严格保留其造型、配色与包装文字不变。" % imgs


def _character_prompt(ch: "dict[str, Any]") -> str:
    """人物设定图描述。只喂外观，五官交给线稿，避免 t2i 画出真人脸后被风控拦。"""
    bits = [ch.get("外观") or "", ch.get("角色") or ""]
    return "，".join(b.strip().rstrip("。") for b in bits if b.strip())


def _scene_prompt(sc: "dict[str, Any]", desc: str = "") -> str:
    # 暗场景（地牢/夜戏）容易生成一张几乎全黑的图，当参考图没有信息量，所以强制曝光下限。
    return ("场景概念图：%s。%s，%s。竖版构图，画面中无人物，4K写实摄影，"
            "整体曝光充足，主体结构与材质清晰可辨，暗部保留细节，无大面积纯黑死黑。"
            % ((desc or sc.get("描述") or sc.get("名称") or "").strip().rstrip("。"),
               sc.get("光线") or "自然光均匀照明", sc.get("色调") or "统一色调"))


# 人物参考图 → 设定图：先图生图把真人脸改成铅笔素描（保脸型五官结构，且能过风控），
# 再按剧本外观换装出全身设定图。素描脸在视频生成阶段由 line_art.REAL_ACTOR_HINT 恢复真人质感。
PERSON_SHEET_PROMPT = (
    "把 @图1 中的人物按以下要求出一张全身人物设定图：严格保持 @图1 人物的面部铅笔素描"
    "（脸型、五官位置与比例、眉眼鼻唇结构）、发型发色与体型完全不变，面部继续保持素描材质，"
    "不要恢复肤色；换装为：%s。正面全身站立，纯色浅灰背景，数码单反相机实拍质感，"
    "影棚柔光箱三点布光；除面部素描外，头发、皮肤、服装、鞋子均为真实照片材质。"
)


def gen_assets(script: "dict[str, Any]", outdir: str,
               product: "Optional[dict[str, Any]]" = None,
               persons: "Optional[list[dict[str, Any]]]" = None) -> "dict[str, Any]":
    """生成人物设定图（线稿脸）与场景概念图，落地到 outdir，返回 {"人物":[],"场景":[]}。

    人物图必须走 line_art：seedance 拒收含真人人脸的参考图，线稿脸既能过风控，
    又保住发型/服装/体型这些跨镜头一致性信息。

    给了 persons（人物参考图）时，主角（依次往后的角色）改走「素描脸设定图」：
    真人照片先图生图转素描脸保住五官结构，再按剧本外观换装——比纯文字线稿多保住
    「是同一个人」的身份信息。失败（如风控拒图）退回文字线稿，不打断流水线。

    给了 product 且这张图里会出现商品时，走参考图生图：商品原图进 ref_images，
    提示词里用 @图N 锚点指到它，避免文生图自己编一个假商品被 seedance 照抄。
    """
    os.makedirs(outdir, exist_ok=True)
    prod_refs = list((product or {}).get("image_urls") or (product or {}).get("images") or [])
    plan = plan_asset_refs(script, product) if prod_refs else {}
    declare = _ref_declare(len(prod_refs)) if prod_refs else ""
    # 人物参考按「主角优先，其余按剧本顺序」依次分配给角色
    ordered = sorted(script.get("人物设定") or [],
                     key=lambda c: 0 if "主角" in str(c.get("角色") or "") else 1)
    assign = {str(c.get("编号")): p for c, p in zip(ordered, persons or [])}

    def decide(it: "dict[str, Any]") -> "tuple[bool, str, dict[str, Any]]":
        """返回 (是否带商品参考图, LLM 改写的提示词, 记到产物里的判定信息)。"""
        p = plan.get(str(it.get("编号"))) or {}
        hit = bool(p.get("出现商品")) and bool(prod_refs)
        return hit, (p.get("提示词") or "") if hit else "", {
            "出现商品": bool(p.get("出现商品")), "判定来源": p.get("判定来源") or "未判定"}

    def with_check(gen, rec: "dict[str, Any]", plain: str) -> str:
        """带商品参考图出图 + 商品一致性核验，返回最终采用的图 url。

        出一张验一张：不过就把走样点写进提示词重来一次，仍不过就退回 plain（不带商品参考图，
        提示词里也不能再留 @图N 引用）。宁可设定图里没有商品——商品会作为独立参考图另行下发，
        而一台印错字、混了配色的假商品进了锚点，后面每一段都会照抄它。
        """
        url = gen(rec["提示词"], rec["ref_images"])
        checks = [_check_asset_product(rec["ref_images"][0], url)]
        if checks[0]["一致"] is False:
            retry = rec["提示词"] + ASSET_AVOID_HINT % checks[0]["问题"]
            url2 = gen(retry, rec["ref_images"])
            checks.append(_check_asset_product(rec["ref_images"][0], url2))
            if checks[-1]["一致"] is False:
                rec.update({"提示词": plain, "ref_images": [],
                            "商品参考图退回": checks[-1]["问题"]})
                url = gen(plain, [])
            else:
                rec["提示词"], url = retry, url2
        rec["商品核验"] = checks
        return url

    def gen_char(ch: "dict[str, Any]") -> "dict[str, Any]":
        rec: "dict[str, Any]" = {"编号": ch.get("编号"), "姓名": ch.get("姓名")}
        hit, rewritten, info = decide(ch)
        rec.update(info)
        desc = rewritten or _character_prompt(ch)
        rec["提示词"] = (declare + desc) if hit else desc
        rec["ref_images"] = prod_refs if hit else []
        person = assign.get(str(ch.get("编号")))
        if person:
            try:
                # 用户传了人物参考图：先转线稿脸（真人脸会被 seedance 风控拒），再按外观出设定图。
                # 商品名与商品图一起带上：线稿化是「读图出文字→文字生图」，人物身上的商品
                # 在文字这一步被叫错品类就会一路错下去（见 line_art._DESCRIBE_PRODUCT）。
                sketch = line_art.to_line_art(
                    person["url"],
                    product="、".join(x for x in ((product or {}).get("name"),
                                                 (product or {}).get("appearance")) if x),
                    product_refs=prod_refs[:2])
                rec["素描图"] = sketch
                rec["提示词"] = PERSON_SHEET_PROMPT % (ch.get("外观") or person.get("desc") or "")
                url = aigc.gen_image(rec["提示词"], ref_images=[sketch])
                rec.update({"url": url, "人物参考": person.get("image"),
                            "ref_images": [sketch], "mode": "person_sketch",
                            "file": storage.download(url, os.path.join(
                                outdir, "char_%s.jpg" % ch.get("编号")))})
                return rec
            except Exception as exc:  # noqa: BLE001
                # 常见于真人风控拒图：退回文字线稿，人物身份靠外观描述近似
                rec["人物参考回退"] = str(exc)[:300]
                rec["提示词"] = (declare + desc) if hit else desc
                rec["ref_images"] = prod_refs if hit else []
        try:
            def gen(text: str, refs: list) -> str:
                return line_art.gen_line_art_frame(text, ref_images=refs or None)

            # 退回时用不带 @图N 引用的原始描述：@图N 指到一批没下发的图，模型只会自己编
            url = (with_check(gen, rec, _character_prompt(ch))
                   if rec["ref_images"] else gen(rec["提示词"], []))
            rec["url"] = url
            rec["file"] = storage.download(url,
                                           os.path.join(outdir, "char_%s.jpg" % ch.get("编号")))
            rec["mode"] = "line_art+商品参考图" if rec["ref_images"] else "line_art"
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:300]
        return rec

    def gen_scene(sc: "dict[str, Any]") -> "dict[str, Any]":
        rec: "dict[str, Any]" = {"编号": sc.get("编号"), "名称": sc.get("名称")}
        hit, rewritten, info = decide(sc)
        rec.update(info)
        rec["提示词"] = (declare if hit else "") + _scene_prompt(sc, rewritten)
        rec["ref_images"] = prod_refs if hit else []
        try:
            def gen(text: str, refs: list) -> str:
                return aigc.gen_image(text, ref_images=refs or None)

            url = (with_check(gen, rec, _scene_prompt(sc))
                   if rec["ref_images"] else gen(rec["提示词"], []))
            rec["url"] = url
            rec["file"] = storage.download(url,
                                           os.path.join(outdir, "scene_%s.jpg" % sc.get("编号")))
            rec["mode"] = "t2i+商品参考图" if rec["ref_images"] else "t2i"
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:300]
        return rec

    chars = script.get("人物设定") or []
    scenes = script.get("场景设定") or []
    with cf.ThreadPoolExecutor(4) as ex:
        char_recs = list(ex.map(gen_char, chars))
        scene_recs = list(ex.map(gen_scene, scenes))
    return {"人物": char_recs, "场景": scene_recs}


# ---------------- 第 3 步：≤15s 分段 ----------------
def _is_hard_cut(shot: "dict[str, Any]") -> bool:
    """这一镜的入点是否为镜头切换。切换处断段拼接才不会有不和谐感。"""
    t = str(shot.get("转场") or "")
    if any(k in t for k in SOFT_TRANSITIONS):
        return False
    return any(k in t for k in HARD_TRANSITIONS) or not t.strip()


def _dur(shot: "dict[str, Any]") -> float:
    try:
        return max(0.1, float(shot.get("时长秒") or 0))
    except (TypeError, ValueError):
        return 2.0


def _split_long_shot(shot: "dict[str, Any]") -> "list[dict[str, Any]]":
    """单镜就超过 15s：只能在同一镜内部切开，标记需要桥接帧。

    台词必须按句分给各段，不能 dict(shot) 整份复制：分段是各自独立生成的，
    每段都会被下发「人物必须在本镜中清晰说出这句台词」（见 produce_video 的段提示词），
    复制一份就等于让同一句话在成片里念两遍。切不出足够份数的，后面几段留空当纯画面续拍。
    """
    total = _dur(shot)
    n = int(total // MAX_SEG) + (1 if total % MAX_SEG else 0)
    part = total / n
    text = str(shot.get("台词") or "").strip()
    parts = [""] * n
    if text and n > 1:
        got = _deal_sentences(_sentences(text), n)
        if len(got) < n:
            got = _deal_sentences(_sentences(text, with_comma=True), n)
        parts = (got + [""] * n)[:n]
    elif text:
        parts = [text]
    out: "list[dict[str, Any]]" = []
    for i in range(n):
        s = dict(shot)
        s["时长秒"] = round(part, 1)
        s["台词"] = parts[i]
        s["_分段"] = "%d/%d" % (i + 1, n)
        s["转场"] = shot.get("转场") if i == 0 else "同镜续拍"
        out.append(s)
    return out


def _anchor_text(anchors: "Optional[list[dict[str, Any]]]",
                 shot_nos: "Optional[list]" = None) -> str:
    """把剧本「视觉锚点」渲染成一句话，随每个分段下发。

    分段是各自独立生成的：锚点不随段下发，同一个道具/光效/屏幕画面在不同段里
    就会各画各的颜色（case 17：手里的发光软体是绿色，后一段屏幕壁纸却变成蓝色）。

    shot_nos 给定时只留「出现镜头」与这一段有交集的锚点：多配色商品会各自成为一条锚点
    （冷烟紫机、森野绿机），全部塞进每一段等于要求这一段同时画出两种配色，模型只会混搭。
    锚点没写「出现镜头」的按全片通用处理，照旧每段都带。
    """
    want = {str(n) for n in (shot_nos or [])}
    items = []
    for a in anchors or []:
        if not (isinstance(a, dict) and a.get("名称") and a.get("形态与颜色")):
            continue
        at = [str(n) for n in (a.get("出现镜头") or [])]
        if want and at and not (want & set(at)):
            continue
        items.append("%s＝%s" % (a.get("名称"), a.get("形态与颜色")))
    if not items:
        return ""
    return "全片视觉锚点（每段画面里这些元素的形态与颜色必须与此完全一致）：" + "；".join(items) + "。"


def plan_segments(shots: "list[dict[str, Any]]",
                  anchors: "Optional[list[dict[str, Any]]]" = None) -> "list[dict[str, Any]]":
    """把分镜打包成 ≤15s 的片段。anchors 为剧本「视觉锚点」，渲染后随每段下发。

    动态规划最小化代价：段数 + 在连续画面处断开的惩罚 + 段过短的惩罚。
    切点只允许落在镜头边界；单镜超 15s 时才在镜内切，并标记需要桥接帧。
    锚点按段过滤（见 _anchor_text）：每段只带这一段镜头里真会出现的那几条。
    """
    flat: "list[dict[str, Any]]" = []
    for s in shots:
        flat.extend(_split_long_shot(s) if _dur(s) > MAX_SEG else [s])
    n = len(flat)
    if not n:
        return []

    pre = [0.0]
    for s in flat:
        pre.append(pre[-1] + _dur(s))

    INF = float("inf")
    best = [INF] * (n + 1)
    prev = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j):
            dur = pre[j] - pre[i]
            if dur > MAX_SEG:
                continue
            cost = best[i] + 1.0
            if cost == INF:
                continue
            if dur < MIN_SEG:                       # 会被 clamp 到 4s，画面得拉长
                cost += 3.0 * (MIN_SEG - dur)
            if i > 0 and not _is_hard_cut(flat[i]):  # 在连续画面处断开
                cost += 8.0
            if j < n and not _is_hard_cut(flat[j]):
                cost += 8.0
            cost += 0.3 * (MAX_SEG - dur)            # 尽量把段填满，减少段数
            if cost < best[j]:
                best[j] = cost
                prev[j] = i

    bounds, j = [], n
    while j > 0:
        bounds.append((prev[j], j))
        j = prev[j]
    bounds.reverse()

    segments = []
    for k, (i, j) in enumerate(bounds):
        group = flat[i:j]
        raw = pre[j] - pre[i]
        head = group[0]
        # 锚点按段过滤：这一段只带这一段镜头里真会出现的锚点
        nos = [s.get("序号") for s in group]
        note = _anchor_text(anchors, nos)
        if k == 0:
            cont = "开场"
        elif head.get("转场") == "同镜续拍":
            cont = "同镜续拍：需用 line_art.gen_line_art_frame 按上一段末画面生成桥接首帧"
        elif _is_hard_cut(head):
            cont = "镜头切换：直接硬切衔接，无需桥接帧"
        else:
            cont = "连续画面（%s）：单镜时长限制被迫在此断开，建议生成桥接首帧" % head.get("转场")
        segments.append({
            **({"视觉锚点": note} if note else {}),
            "段号": k + 1,
            "镜头序号": nos,
            "镜内分段": [s["_分段"] for s in group if s.get("_分段")],
            "原始时长秒": round(raw, 1),
            "生成时长秒": int(max(MIN_SEG, min(MAX_SEG, round(raw)))),
            "衔接": cont,
            "场景编号": sorted({s.get("场景编号") for s in group if s.get("场景编号")}),
            "人物编号": sorted({c for s in group for c in (s.get("人物编号") or [])}),
            "需要模特出镜": _need_model(group),
            "视频提示词": _seg_prompt(group),
            "台词": [s.get("台词") for s in group if s.get("台词")],
        })
    return segments


def _need_model(group: "list[dict[str, Any]]") -> "Optional[bool]":
    """这一段要不要模特出镜：任一镜要 → 要；全部明确不要 → 不要；否则 None（未判定）。"""
    flags = [s.get("需要模特出镜") for s in group]
    if any(f is True for f in flags):
        return True
    if flags and all(f is False for f in flags):
        return False
    return None


def _seg_prompt(group: "list[dict[str, Any]]") -> str:
    """把段内各镜的视频提示词串成一条 seedance 提示词，镜间明确写出切换方式。

    镜级视频提示词自身已带运镜与景别（见输出结构里的字段说明），这里不再重复前缀。
    """
    parts: "list[str]" = []
    for idx, s in enumerate(group):
        body = (s.get("视频提示词") or s.get("画面") or "").strip().rstrip("。")
        if not body:
            body = "%s，%s" % (s.get("景别") or "中景", s.get("运镜") or "固定机位")
        if idx:
            parts.append("随后%s切换，%s" % (s.get("转场") or "硬", body))
        else:
            parts.append(body)
    return "。".join(parts) + "。"


# ---------------- 输出 ----------------
def _fmt(v: Any) -> str:
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            return "\n" + "\n".join("  - " + "；".join("%s: %s" % (k, x[k]) for k in x) for x in v)
        return "、".join(str(x) for x in v) or "-"
    if isinstance(v, dict):
        return "\n" + "\n".join("  - %s: %s" % (k, _fmt(x)) for k, x in v.items())
    return str(v)


def to_markdown(rec: "dict[str, Any]") -> str:
    sc = rec["剧本"]
    lines = ["# %s" % (sc.get("剧本名") or "仿写剧本"), "",
             "> 参考爆款：%s" % rec["参考片"], "", "## 创意与结构", ""]
    for k in ("一句话概要", "核心创意", "开场钩子", "结构", "音乐", "情绪曲线", "节奏"):
        if k in sc:
            lines.append("- **%s**：%s" % (k, _fmt(sc[k])))

    lines += ["", "## 如何复刻的", ""]
    for r in sc.get("复刻说明") or []:
        lines.append("### %s" % r.get("维度"))
        lines.append("- 参考片：%s" % r.get("参考片"))
        lines.append("- 本片：%s" % r.get("本片"))
        lines.append("- 原理：%s" % r.get("原理"))
        lines.append("")

    lines += ["## 人物设定", ""]
    amap = {c.get("编号"): c for c in (rec.get("素材") or {}).get("人物") or []}
    for c in sc.get("人物设定") or []:
        lines.append("### %s（%s，%s）" % (c.get("姓名"), c.get("编号"), c.get("角色")))
        lines.append("- 外观：%s" % c.get("外观"))
        lines.append("- 性格：%s" % c.get("性格"))
        lines.append("- 转变：%s" % c.get("转变"))
        a = amap.get(c.get("编号")) or {}
        lines.append("- 设定图：%s" % (a.get("file") or a.get("error") or "未生成"))
        lines.append("")

    lines += ["## 场景设定", ""]
    smap = {s.get("编号"): s for s in (rec.get("素材") or {}).get("场景") or []}
    for s in sc.get("场景设定") or []:
        lines.append("### %s（%s）" % (s.get("名称"), s.get("编号")))
        lines.append("- 描述：%s" % s.get("描述"))
        lines.append("- 光线：%s ｜ 色调：%s" % (s.get("光线"), s.get("色调")))
        a = smap.get(s.get("编号")) or {}
        lines.append("- 概念图：%s" % (a.get("file") or a.get("error") or "未生成"))
        lines.append("")

    lines += ["## 分镜", ""]
    for s in sc.get("分镜") or []:
        lines.append("### 镜 %s（对应参考第 %s 镜，%ss）" %
                     (s.get("序号"), s.get("对应参考镜"), s.get("时长秒")))
        for k in ("景别", "运镜", "场景编号", "人物编号", "画面", "动作", "台词", "特效",
                  "转场", "音效音乐", "叙事功能", "需要模特出镜", "视频提示词"):
            if k in s:
                lines.append("- **%s**：%s" % (k, _fmt(s[k])))
        lines.append("")

    lines += ["## 生成分段（单段 ≤15s）", ""]
    for g in rec.get("分段") or []:
        lines.append("### 段 %s ｜ 镜 %s ｜ 原始 %ss → 生成 %ss" %
                     (g["段号"], g["镜头序号"], g["原始时长秒"], g["生成时长秒"]))
        lines.append("- 衔接：%s" % g["衔接"])
        lines.append("- 参考图：人物 %s ｜ 场景 %s" % (g["人物编号"], g["场景编号"]))
        lines.append("- 需要模特出镜：%s"
                     % {True: "是", False: "否"}.get(g.get("需要模特出镜"), "未判定"))
        lines.append("- 提示词：%s" % g["视频提示词"])
        if g["台词"]:
            lines.append("- 台词：%s" % "／".join(g["台词"]))
        lines.append("")
    return "\n".join(lines)


def build(ref_json: str, product: "dict[str, Any]", with_images: bool = True,
          mode: str = "creative", outdir: str = "",
          person_images: "Optional[list[str]]" = None,
          narrative_gate: bool = True) -> "dict[str, Any]":
    """完整流水线，返回 {"参考片","商品","剧本","素材","分段"}。mode 见 write_script()。

    outdir 给定时设定图落在该目录的 assets/ 下（任务化调用用），否则按商品名落在 OUT_ROOT。
    person_images 为用户上传的人物参考图：主角（依次往后）替换成参考图里的人。

    顺序是刻意的：写剧本 → 两层剧本门禁（逐镜审查 + 整片叙事审查/修补/复审）→ 设定图 → 分段。
    门禁跑在任何「生成」动作之前——设定图是第一次真的花钱出图，分段之后就是 seedance 出片，
    等到那时候再发现整片看不懂已经晚了（钱花完、时间花完）。

    narrative_gate=True 时，整片叙事审查复审后仍判「可理解性不足」就**就此停下**：
    不出设定图、不分段，返回的 built 带 "拦下" 标记，由调用方决定怎么报错。
    剧本本身照旧返回，方便落盘给人看审查与修补痕迹。
    """
    with open(ref_json, encoding="utf-8") as fh:
        ref = json.load(fh)

    product = dict(product)
    if product.get("images"):
        product["image_descriptions"] = [d["desc"] for d in describe_products(product["images"])]

    persons = describe_persons(person_images)
    script = write_script(ref, product, mode=mode, persons=persons)
    name = product.get("name") or "product"
    outdir = outdir or os.path.join(OUT_ROOT, "".join(c for c in name if c not in '/\\:*?"<>|'))
    base = {"参考片": ref.get("视频") or ref.get("video") or os.path.basename(ref_json),
            "商品": product, "剧本": script,
            "人物参考": [{"image": p.get("image"), "desc": p.get("desc")} for p in persons],
            "outdir": outdir}
    tale = script.get("叙事连贯审查") or {}
    if narrative_gate and tale.get("状态") == "可理解性不足":
        tale["拦下"] = True
        return dict(base, 素材={"人物": [], "场景": []}, 分段=[],
                    拦下="整片叙事审查不通过：%s（可理解性 %s，盲测看成「%s」）"
                         % (tale.get("总体问题") or "见叙事连贯审查",
                            tale.get("可理解性", "-"), tale.get("盲测概要") or "-"))
    assets = (gen_assets(script, os.path.join(outdir, "assets"), product=product,
                         persons=persons)
              if with_images else {"人物": [], "场景": []})
    return dict(base, 素材=assets,
                分段=plan_segments(script.get("分镜") or [], script.get("视觉锚点") or None))


def main():
    # ==== 调试参数：直接改这里 ====
    REF_JSON = "output/script_analysis/10_武侠_参考_平底锅.json"  # 参考爆款的拆解结果
    PRODUCT = {                                              # 商品信息，字段都可缺省
        "name": "格力锅铲",
        "category": "厨房用具/锅铲",
        "selling_points": ["柔韧哑光铲头，翻炒顺滑，保护不粘锅涂层不刮花",
                           "铲面一字型镂空，翻炒盛菜顺手滤掉多余油脂",
                           "不锈钢流线手柄，耐腐耐用、握感舒适好清洗",
                           "手柄尾部圆形挂孔，洗完直接挂起收纳不占地"],
        "copy": "柔韧铲头不伤锅，格力品质好锅铲",
        "audience": "家庭主妇、独居青年等注重厨具实用性的做饭人群",
        "appearance": "黑银拼色，黑色哑光铲头带一字镂空，银色不锈钢手柄正面印黑色「格力」字样，尾部带挂孔",
        "usage_scene": "家庭日常厨房烹饪，尤其适合搭配不粘锅炒菜、煎蛋煎鱼与沥油滤汤",
        # 商品图，本地路径或公网 URL；正式链路由 produce_video 从 materials/{编号} 自动取
        "images": ["/root/jmzhang/baidu/ViralForge/materials/10/格力锅铲.jpeg"],
    }
    WITH_IMAGES = True    # 是否生成人物/场景设定图（关掉可省 t2i 费用，只出剧本和分段）
    WITH_MARKDOWN = True  # 是否额外导出便于阅读的 script.md
    # ==============================

    here = os.path.dirname(os.path.abspath(__file__))
    ref_json = REF_JSON if os.path.isabs(REF_JSON) else os.path.join(here, REF_JSON)
    if not os.path.isfile(ref_json):
        print("参考拆解 json 不存在:", ref_json)
        return 1

    print("仿写剧本中（参考 %s）..." % os.path.basename(ref_json))
    rec = build(ref_json, PRODUCT, with_images=WITH_IMAGES)

    outdir = rec.pop("outdir")
    os.makedirs(outdir, exist_ok=True)
    jp = os.path.join(outdir, "script.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    print("分镜 %d 个，分段 %d 段" % (len(rec["剧本"].get("分镜") or []), len(rec["分段"])))
    print("产物:\n  %s" % jp)
    if WITH_MARKDOWN:
        mp = os.path.join(outdir, "script.md")
        with open(mp, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(rec))
        print("  %s" % mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
