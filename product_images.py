"""商品参考图链路：结构化理解 → 选图 → 编辑改造 → 分镜级分配 → 中间状态图。

固定顺序（细节见 Skills/ProductReferenceImagePlanner.md）：
1. 每张商家图一次结构化理解（caption / 展示信息 / 视角 / 完整性 / 干扰），落 image_index.json
2. 只读这份理解挑基准参考图并排序，@图片1 必须是完整全貌
3. 选中的图判断要不要编辑；要编辑的由 VLM 一次写好指令，一张一张生成、每张都与原图比对
4. 剧本出来后按镜分配参考图，并为视频模型画不准的关键状态生成中间状态图
用户没传商品图时走 harvest_product_images：从素材抽帧 + 据此生成白底商品图。
"""
import concurrent.futures as cf
import json
import os

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from media import extract_frame, _sec
from task_store import _p, _rel, log


# ---------------- 步骤 4：商品事实卡 ----------------
# 没有商品图时，从用户素材里把商品抠出来当参考图：挑片段 → 抽帧 → 判清晰度 → 生白底图。
# 下游（write_script.gen_assets、_gen_piece）都是读 fact_card 的 image_urls，补齐后自动生效。
PICK_FRAME_PROMPT = """下面是用户素材视频切好的片段清单。用户没有上传商品图，
我要从素材里抽一帧当商品参考图。

挑出商品露出最完整、最清晰的 1-3 个片段，按可用性从高到低排。
优先特写或近景、商品居中、没有手部或其它物体大面积遮挡的片段。
**商品必须保持它原本的形状**：被手指捏压、揉搓成条、拉伸、折弯而变形的时间点一律不要选，
宁可选商品静静躺着、被托着或摆在桌面上的时间点——哪怕那个点稍微远一点、糊一点。
形变比遮挡和清晰度更该排除：糊一点只是细节少，形状被捏变了会让整片的商品都是变形的。
「抽帧秒数」给这个片段内商品最清楚**且没有变形**的时间点，是相对整条视频的绝对秒数，
要落在片段起止之间。

【商品】名称：__NAME__，品类：__CATEGORY__
（名称或品类为空时，就按素材里反复出现的那个主要商品来理解）

【片段】
__SEGMENTS__

输出 json：{"候选": [{"片段ID": "", "抽帧秒数": 0.0, "理由": ""}]}
只输出 json。"""

FRAME_JUDGE_PROMPT = """这是从视频里抽出来的一帧，我要拿它当商品参考图用。判断并输出 json：

{"商品": "画面里的主体商品是什么",
 "清晰": true/false,
 "形变": "商品有没有偏离它原本的形状：没有就写「无」；有就写清是怎么变的，如 被手指捏扁 / 被揉搓成细长条 / 被拉伸 / 被折弯 / 被按压凹陷",
 "问题": "遮挡/模糊/太小/有手或人/背景杂乱/商品被捏压变形，都没有就写空字符串",
 "外观": "外形结构、材质质感、主色配色、包装上的显著文字或图案，60字内",
 "白底图": [{"要表现什么": "如 正面全貌 / 侧面 / 背面 / 打开后露出内容物"}],
 "卖点": ["从画面能看出来的卖点，看不出就给空数组"]}

「形变」只看商品自身的形状有没有被改变：手托着、手指捏着但商品没变形，写「无」；
商品被握变形、揉搓成条、压扁、拉长、折弯才算有形变。可折叠/可开合商品按设计正常
开合（翻盖打开、折叠屏展开）不是形变，写「无」。
「外观」写商品原本的形状，不要把被捏出来的临时形状当成它的设计。

「白底图」只能提出这一帧里实际看得见的角度或状态：帧里只拍到正面就只给正面，
背面/内部结构这类帧里看不到的不要提——生成模型会凭空编造出假商品。
比如帧里是奶粉罐正面就给正面全貌；帧里恰好拍到打开状态才给露出粉末的图。
1 张够就给 1 张，最多 3 张。只输出 json。"""

WHITE_BG_PROMPT = ("纯白背景商品主图：%s。这一张要表现%s。"
                   "严格保留参考图中商品的外形结构、材质质感、配色与包装文字，不要改设计，"
                   "商品居中完整入画，影棚柔和均匀光，轻微接地阴影，"
                   "画面里只有商品本身，没有手、没有人、没有其它道具与背景陈设，4K 电商主图。")

MAX_WHITE_BG = 3            # 白底图最多生成几张，控成本
MAX_PICK_SEGMENTS = 80      # 喂给挑片段模型的片段上限
MAX_STATE_LOOKUP = 6        # 核验「素材里有没有这个状态」时一次最多打开几张候选图

# 「形变」字段判为「没有变形」的写法。模型不会每次都老实回「无」，也会写「无形变」
# 「没有明显变形」这类，所以按包含匹配收口，别让一句「无明显形变」被当成有形变。
# 上传图（IMAGE_INDEX_PROMPT）与抽帧（FRAME_JUDGE_PROMPT）两条链路共用这一套判定口径。
NO_DEFORM_WORDS = ("无", "否", "没有", "未变形", "不存在")


def _is_deformed(value) -> bool:
    """这张图/帧里的商品是不是被人为挤压按压揉搓拉伸而偏离了原本形状。

    判不出（字段缺失或空）一律按「没变形」处理：宁可放过一张，也不要因为字段没填
    就把唯一可用的商品参考图判死（没有商品图时生成端会凭空编造一个假商品）。
    """
    text = str(value or "").strip()
    return bool(text) and not any(w in text for w in NO_DEFORM_WORDS)

# 商家给的图片素材质量参差：有整体全貌、有局部特写、有大字报海报。挑参考图之前先逐张做
# 结构化理解（产物 product/image_index.json），后面选图、判断要不要编辑都读这份结构，
# 不再靠文件顺序猜——case 17 的 @图片1 是镜头模组局部特写，生成端照着它编出了假手机。
IMAGE_INDEX_PROMPT = """这是一张商品图片素材，后面要拿去当 AI 生图/生视频的商品参考图。
结构化理解这一张图，输出 json：

{"caption": "一句话说清这张图在拍什么，20-40字",
 "商品": "画面主体商品是什么",
 "展示信息": ["这张图能看清商品的哪些信息，逐条写，如 正面全貌 / 机身厚度 / 摄像头模组排布 / 屏幕显示内容 / 包装印刷"],
 "可见状态": ["画面里这个商品正处在什么状态，逐条把状态本身写出来；没什么可说的就给空数组"],
 "形变": "商品有没有偏离它原本的形状：没有就写「无」；有就写清是怎么变的，如 被手指捏扁 / 被揉搓成细长条 / 被拉伸 / 被折弯 / 被按压凹陷",
 "视角或状态": "正面 / 背面 / 侧面 / 45度 / 俯视 / 打开状态 / 使用中 等",
 "是否局部细节": true/false,
 "完整性": "完整 / 基本完整 / 局部",
 "完整展示的角度": "这张图完整展示了哪个角度的全貌（如 背面全貌）；只是局部特写就写空字符串",
 "商品占比": "大 / 中 / 小",
 "同图商品数": 画面里出现了几个该商品（整数）,
 "干扰": ["压在画面上的营销大字、促销贴片、水印、第三方logo、人、手、杂乱背景，逐个写清是什么、在画面哪里；没有就给空数组"],
 "画质": "清晰 / 一般 / 模糊"}

商品包装或机身自身印刷的品牌名、产品名属于商品本身，不算干扰。

「形变」只看商品自身的形状有没有被改变：手托着、手指捏着但商品没变形，写「无」；
商品被握变形、揉搓成条、压扁、拉长、折弯才算有形变。可折叠/可开合商品按设计正常
开合（翻盖耳机仓打开、折叠屏展开）不是形变，写「无」。

「展示信息」「可见状态」写具体内容，不要只给类别名：写「背部副屏亮起显示紫底白色指针模拟时钟」，
不要写「屏幕显示内容」；写「翻盖开到约120度，仓内两只耳机都在位」，不要写「打开状态」。
下游挑图、判断某个状态素材里有没有，读的都是这份文字理解而不是原图——归纳成类别名，
这些细节就永久丢了，后面只会误判成「素材里没有」。只输出 json。"""

SELECT_REFS_PROMPT = """给「__NAME__」挑商品参考图。下面是每张候选图的结构化理解：

__INDEX__

挑选口径：
- 用尽可能少的图覆盖商品完整外观：完整全貌 > 其它角度/状态 > 关键局部细节
- 排第 1 的必须是最能代表商品整体的完整全貌图——生成端把第 1 张当主锚点，
  第 1 张是局部特写就会被编造出一个假商品
- 排第 1 的还必须是「同图商品数」为 1 的图：一张图里几台不同配色/款式同框时，
  生成端分不清该照哪一台画，主锚点自带多种配色等于要求它混搭。多款同框的图可以选中，
  但只能排在后面当配色/角度对照，不能当第 1 张
- 排第 1 的还必须是「形变」为「无」的图：商品被人为挤压、按压、揉搓、拉伸、折弯而
  偏离原本形状时，拿它当主锚点，生成端就会照着那个变形的形状画，整片的商品都是变形的。
  这类图可以选中，但只能排在后面当使用状态对照，不能当第 1 张
- 同一个视角只留信息最全的那张，其余落选
- 素材里有多款 / 多配色 / 多套搭配时，每一款各留一张最有代表性的，不要当成同角度冗余合并掉
- 局部特写只在它提供了别的图没有的关键信息时才留
- 有干扰（营销大字/水印/人手）不是落选理由，选中后会走图像编辑改造；
  画质模糊、或商品占比小且有更好替代的，才落选
- 最多选 __MAX__ 张，信息够了就停（这是事实卡里的基准参考图；落选的图后面还能按分镜取用）

输出 json：
{"选中": [{"编号": 1, "角色": "商品整体 / 背面结构 / 关键细节 等", "理由": "30字内"}],
 "落选": [{"编号": 2, "原因": "与编号1同视角且信息更少"}],
 "缺失": ["候选里完全没有的角度或状态，如 背面全貌；没有就给空数组"]}

「选中」按重要性从高到低排，第 1 个就是 @图片1。只输出 json。"""

# 用户上传的商品图不能直接采信：营销图常见商品主体小、促销大字、牛皮癣贴片、水印。
# 编辑指令由 VLM 一次写完（格式见 Skills/ProductImageEditSkill.md）：只有看得见画面的
# 模型才知道上面压着哪几个字、贴片在哪个角、有几只手，代码套模板只能笼统说
# 「去掉营销文字」，点不到具体元素。
EDIT_PROMPT_PROMPT = """这是一张商品图，要拿它当 AI 生图/生视频的商品参考图。
先判断能不能直接用，不能直接用就写一条图像编辑指令。__HINT__输出 json：

{"商品": "主体商品的通用名，两三个字，如 手机 / 牙刷 / 奶粉罐",
 "可直接用": true/false,
 "问题": "商品主体太小/营销大字/牛皮癣贴片/水印logo/背景杂乱/有手或人/模糊，都没有就写空字符串",
 "改写prompt": "图像编辑指令；可直接用时给空字符串",
 "外观": "商品的外形结构、材质质感、主色配色、包装上的显著文字或图案，60字内"}

「改写prompt」照下面这个格式写，按画面实际情况增删句子：

「手机」保持不变，删除顶部「xiaomi 17 Pro Max」大字、删除右上角「小米徕卡联合研发」红标、\
删除中部宣传语「多一面 更精彩」、删除右下角半透明水印。删除人物。修改为白底图，\
「手机」占据主要画面。删除右边多余的「手机」，删除手。生成商品主图，「手机」居于正中，放大。

写的时候必须做到：
- 商品一律用「」括起来的通用名指代，整句统一，第一句就是「商品」保持不变
- 画面上叠加的文字要把原文抄进指令里（实在看不清就写清位置+颜色+大致内容），
  牛皮癣贴片、水印、logo、二维码同样一个个点名，写清在画面哪个位置
- 商品包装自身印刷的品牌名、产品名属于商品，要写一句「保持包装上印刷的文字不变」，不许删
- 画面里没有的不要写（没有手就别写删除手，只有一个商品就别写删除右边的）
- 同一商品多款/多配色/多件同框（同图商品数≥2）是硬规则：必须只保留最能代表商品的一款，
  其余每一件都用「方位+颜色或特征」单独点名删除（如 删除左边蓝色「手机」、
  删除右上角绿色「手机」），成图只有一个商品主体，配色双拼展示也一样要拆；
  唯一例外：提示里说明这张图的用途就是展示多款/多配色时，所有款式保留，只删干扰元素
- 只做删除干扰元素、换纯白背景、放大居中，不要要求换拍摄角度、换姿态、补拍背面或内部结构
只输出 json。"""

REFINE_BATCH = 1        # 每轮生成几张候选：一张一张跑，出一张就立刻验一张
REFINE_ROUNDS = 3       # 最多几轮；轮次用完仍不过关就退回原图

# 改造是还原不是创作：出一张就和原图对一遍，不过关就把走样点写进指令再出一张——
# 商品本身变了的改造图比原图更糟，假商品会被下游生视频原样照抄。
PICK_REFINED_PROMPT = """第一张图是商品原图，后面 __N__ 张是它的改造图（删掉画面上的营销文字/贴片/水印，换纯白背景）。
从改造图里挑最好的一张（只给了 1 张就判这一张），并判断这张和原图是不是同一个商品，输出 json：

{"最佳": 改造图的序号（1 到 __N__，全都不理想也要挑最接近的那张）,
 "商品一致": true/false,
 "差异": "外形结构、部件、配色、材质、包装印刷文字（品牌名产品名是否被改写或变乱码）哪里变了，没变就写空字符串",
 "理由": "为什么挑这张，30字内"}

挑选口径：商品与原图一致优先，其次干扰元素删得干净、背景纯白、商品大而居中完整。
背景、光线、构图、被删掉的营销贴纸不算差异；按指令删除的多余款式、重复商品也不算差异，
但保留下来的那一款必须与原图中的同款一致。
只有这些才算不一致：商品结构变形、部件增减或挪位、配色改变、材质变了、
在原图里就清晰可读的品牌名或产品名被改写成别的词。
文字一律以原图的可读性为基准：原图里本来就小、本来就糊、本来就看不清的字（认证参数小字、
logo 下方的小行、多款同框时每一款机身上的印字），改造后依旧看不清不算不一致——那是原图
分辨率与商品占比带来的，不是商品变了。
只输出 json。"""

# 中间状态图：视频模型一次画不准的高精度状态（开合、界面显示、指示灯、局部形变）先用
# 图像编辑锁死，再当参考图交给视频模型做状态之间的动态变化。这类图的验收多一条——
# 目标状态必须真的画出来了，只是「商品没变」还不够。
# 装配序列（白板逐个装上零件这类创意）走同一套判定，只是多带一份部件到位清单：清单说
# 「应有」的必须与原图同款，说「应无」的必须真的不在。后者才是主要失败模式——生图模型
# 极爱自作主张把缺的部件补全。普通强一致状态图就是「清单全满、应无为空」的退化情形。
PICK_STATE_PROMPT = """第一张图是商品原图，后面 __N__ 张是把商品改成这个状态后的图：__WANT__
__MANIFEST__
从改造图里挑最好的一张（只给了 1 张就判这一张），输出 json：

{"最佳": 改造图的序号（1 到 __N__，全都不理想也要挑最接近的那张）,
 "商品一致": true/false,
 "状态达成": true/false,
 "差异": "商品本身哪里变了 / 目标状态哪里没做到，都没问题就写空字符串",
 "理由": "为什么挑这张，30字内"}

「商品一致」只看商品本身：外形结构、部件、配色、材质、商品上印刷的品牌名产品名有没有被改写。
目标状态要求的形态变化（打开、展开、点亮、显示内容变化等）本身不算不一致；
背景、光线、构图，以及画面上营销文字/海报排版的变化也都不算——那些不是商品的一部分。
文字一律以原图的可读性为基准：原图里本来就小、本来就糊、本来就看不清的机身小字（认证参数
小字、logo 下方的小行、多款同框时每一款机身上的印字）改造后依旧看不清，同样不算不一致；
只有原图里就清晰可读的品牌标识被改写或糊成乱码才算。
「状态达成」看目标状态是不是真的画出来了，含糊、只改了一半、改错位置（比如改到了别的物体上）都算没达成。
还有一条硬口径：如果这个状态需要原图里根本看不见的结构（展开后的内部、背面、拆开后的部件），
或者要求屏幕/表盘/界面显示出原图里不存在的具体画面（壁纸角色、界面内容、图案、成段文字），
那模型只能凭空编造，一律判 状态达成=false，并在差异里说明是凭空编造。
__MANIFEST_RULE__只输出 json。"""

# 「这个状态素材里到底有没有」只能看图回答。分镜挑参考图那步是纯文字推理（只拿到 image_index
# 的文字理解），它给不出这个结论——文字里没提到不等于素材里没有，实测把「副屏显示指针时钟」
# 这种图5里真实存在的状态判成了「素材里没有该显示内容」，5 张中间状态图全被放弃。
# 所以文字层只负责列候选，最终裁定放在这里，把候选图真的打开看一遍。
STATE_LOOKUP_PROMPT = """下面 __N__ 张图是同一个商品的素材图，按顺序编号 1 到 __N__。
要找的状态：__WANT__

逐张看，判断有没有哪张图里这个状态已经真实存在——是画面里当场就能看见，
不是「相似」、不是「可以推测」、不是「换个角度应该也有」。
显示内容、印刷图案、成段文字这类要照实抄的状态，画面里必须真的有这个内容才算命中。
命中多张就选这个状态拍得最清楚、商品也最完整的那张。

输出 json：{"命中": 命中的图序号，都不算就填 0, "理由": "为什么算命中 / 为什么都不算，40字内"}
只输出 json。"""

# 部件到位清单：装配/拆解序列里，每一张中间图该有哪些部件、该缺哪些部件，由剧本层声明。
# 判定口径整体让位给清单——不能再问「和原图一模一样吗」，那是这类创意必然答不对的问题。
MANIFEST_BLOCK = """
【这张图的部件到位清单】这是一个__NARR__序列的第 __I__/__T__ 张，按清单判，不要按「和原图一模一样」判：
- 应有（必须在画面里，且与原图是同一个商品的同一部件）：__HAVE__
- 应无（必须确实不在画面里）：__NONE__

清单模式下的判定口径：
「状态达成」= 应有的都在 且 应无的都不在 且 没有清单外的部件凭空多出来。
应无的部件只要出现就判 状态达成=false（哪怕画得很淡、很小、只露一角），并在差异里点名是哪一个——
把缺的部件自动补全是这类图最常见的失败，必须拦住。
应无部件被移除后留下的位置，呈平整的同材质表面即可，不算凭空编造、不算不一致。
「商品一致」只看应有的那些部件与原图是不是同款：外形结构、比例、配色、材质、印刷文字。
应无部件的缺失不算不一致；商品整体不完整、看起来像半成品，也不算不一致——那正是要的。
清单没提到的内部结构、背面、界面显示内容仍然不许凭空造，出现了就算清单外多出来的东西。
"""

# 清单和普通状态图的口径必然冲突（一个要部件齐全，一个要部件缺着），冲突时以清单为准。
# 这句只在有清单时才拼进去：没有清单的任务，判定 prompt 与从前一字不差。
MANIFEST_RULE = "上面这些口径里凡与「部件到位清单」冲突的，一律以清单那一段为准。\n"


def _manifest_block(m: dict) -> str:
    """把部件到位清单渲染成判定 prompt 里的那一段；没有清单就返回空行（走普通状态图口径）。"""
    if not m:
        return ""
    return (MANIFEST_BLOCK.replace("__NARR__", m.get("叙事类型") or "装配")
            .replace("__I__", str(m.get("序号") or 1)).replace("__T__", str(m.get("共几步") or 1))
            .replace("__HAVE__", "、".join(m.get("应有") or []) or "（未列）")
            .replace("__NONE__", "、".join(m.get("应无") or []) or "（无，这一张就是完整商品）"))


def _norm_items(v, key: str) -> list:
    """模型给的「对象数组」归一成 dict 列表：裸值按 key 包成 dict，None 与其它类型丢掉。

    prompt 里约定的是 [{"编号": 3, …}] / [{"要表现什么": "正面全貌"}] 这种单字段对象数组，
    模型很爱简写成 [3, 4] / ["正面全貌"]，也可能混进 null。直接 `for x in …: x.get(…)` 会抛
    AttributeError，而这几处的契约都是「模型答坏了只少几张图，不打断流水线」——
    `or [{}]` 只挡得住字段整体缺失，挡不住元素类型不对。
    裸值包成 dict 而不是丢掉：模型简写时给的正是那个唯一字段的值，丢了等于把整份结果作废。
    bool 要单独排除，isinstance(True, int) 在 Python 里成立，不排会让 true 变成编号 1。
    """
    out = []
    for x in v or []:
        if isinstance(x, dict):
            out.append(x)
        elif isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip():
            out.append({key: x})
    return out


def _pick_product_segments(rec: dict, segs: list) -> list:
    """让模型挑出商品露出最清楚的片段与抽帧时间点。失败就退回前 3 个片段的中点。"""
    brief = [{k: s.get(k) for k in ("片段ID", "开始时间", "结束时间", "画面", "主体", "景别运镜")
              if s.get(k)} for s in segs[:MAX_PICK_SEGMENTS]]
    try:
        raw = aigc.understand(
            PICK_FRAME_PROMPT.replace("__NAME__", rec["product"].get("name") or "（未填）")
                             .replace("__CATEGORY__", rec["product"].get("category") or "（未填）")
                             .replace("__SEGMENTS__", json.dumps(brief, ensure_ascii=False, indent=1)),
            max_tokens=2048, json_mode=True)
        picks = write_script._parse_json(raw).get("候选") or []
    except Exception as exc:  # noqa: BLE001
        log(rec, "挑片段失败，退回前几个片段：%s" % str(exc)[:120])
        picks = []
    if not picks:
        picks = [{"片段ID": s.get("片段ID"), "抽帧秒数": None, "理由": "兜底：按顺序取前几个片段"}
                 for s in segs[:3]]
    return picks[:3]

def _frame_rank(info: dict) -> tuple:
    """抽帧的优劣排序键（元组越大越好）：不变形排在清晰前面。

    形变优先级高于清晰度：糊一点只是白底图少些细节，形状被捏变了会让整片的商品一直是
    那个变形的形状（实测 c1b4：耳塞被两指捏成尖角的那帧被当成商品参考图进了事实卡）。
    """
    return (0 if _is_deformed(info.get("形变")) else 1, 1 if info.get("清晰") else 0)


def harvest_product_images(rec: dict) -> dict:
    """没有商品图时，从用户素材里抽商品帧，再据此生成白底商品图。

    返回 {"抽帧": [...], "白底图": [...], "判定": {...}, "候选": [...], "最佳帧": ...}，
    任何一步失败都只是少几张图，不抛异常打断整条流水线。
    「抽帧」是全部抽出来的帧（留档用，顺序即候选顺序）；下游要拿去当参考图的是「最佳帧」，
    别用「抽帧」的第一张——候选顺序是模型给的，不代表质量。
    """
    tid = rec["task_id"]
    idx_path = _p(tid, "assets", "material_index.json")
    if not os.path.isfile(idx_path):
        log(rec, "没有素材切片索引，抽不出商品帧（事实卡将没有商品图）")
        return {}
    with open(idx_path, encoding="utf-8") as fh:
        segs = (json.load(fh) or {}).get("片段") or []
    if not segs:
        log(rec, "用户素材没有可用片段，抽不出商品帧（事实卡将没有商品图）")
        return {}

    by_id = {s.get("片段ID"): s for s in segs}
    out: dict = {"抽帧": [], "白底图": [], "判定": {}, "候选": []}
    for i, pick in enumerate(_pick_product_segments(rec, segs), 1):
        seg = by_id.get(pick.get("片段ID")) or {}
        src = seg.get("源文件")
        if not src or not os.path.isfile(src):
            continue
        start, end = _sec(seg.get("开始时间")), _sec(seg.get("结束时间"))
        mid = start + max(0.1, (end - start) / 2)
        try:
            at = float(pick.get("抽帧秒数"))
        except (TypeError, ValueError):
            # 兜底候选（挑片段失败）的「抽帧秒数」是 None，float(None) 抛 TypeError。
            # 这里以前落到 0.0，而首个片段经 analyze_materials._fix_timeline 重排后
            # start 恒为 0.0，`start <= 0.0 <= end` 成立 → 下面的中点兜底不生效，
            # 抽的是整条视频的第 0 帧（短视频首帧多半是黑场/淡入/转场，最不能当商品参考）。
            at = mid
        if not (start <= at <= end):        # 模型给的时间点不在片段里就取中点
            at = mid
        dst = _p(tid, "product", "frames", "f%02d.jpg" % i)
        try:
            extract_frame(src, at, dst)
        except Exception as exc:  # noqa: BLE001
            log(rec, "抽帧失败（%s @%.1fs）：%s" % (os.path.basename(src), at, str(exc)[:120]))
            continue
        pick.update({"源文件": src, "抽帧秒数": round(at, 2), "文件": _rel(tid, dst)})
        out["候选"].append(pick)
        out["抽帧"].append(dst)
        try:
            info = write_script._parse_json(
                aigc.understand(FRAME_JUDGE_PROMPT,
                                media=[{"type": "image", "url": storage.upload(dst)}],
                                max_tokens=1024, json_mode=True))
        except Exception as exc:  # noqa: BLE001
            info = {"清晰": False, "问题": "判定失败：%s" % str(exc)[:120]}
        info["帧文件"] = _rel(tid, dst)
        # 模型的布尔必须过 as_bool：裸真值会把 "false"/"否" 当成清晰，直接 break 在一张
        # 看不清商品的帧上，而这张帧是白底图与事实卡 appearance 的唯一依据。
        clear = rules.as_bool(info.get("清晰")) is True
        info["清晰"] = clear
        info["商品变形"] = _is_deformed(info.get("形变"))
        pick["形变"] = info.get("形变")
        pick["商品变形"] = info["商品变形"]
        if not out["判定"] or _frame_rank(info) > _frame_rank(out["判定"]):
            out["判定"] = info
            out["最佳帧"] = dst
        if _frame_rank(info) == (1, 1):   # 又清晰又没变形，不用再看后面的候选
            break

    frame = out.get("最佳帧")
    if not frame:
        return out
    judge = out["判定"]
    # 全部候选都是变形的：仍然用最好的那一张（没有商品图时生成端会凭空编造假商品，
    # 那比形状不对更糟），但把这件事显式记下来，报告与复核看得见，不静默
    if judge.get("商品变形"):
        log(rec, "抽帧候选里没有商品未变形的帧，只能用变形帧当参考（%s）"
            % str(judge.get("形变"))[:60])
        out["形变告警"] = "全部抽帧候选都是变形的商品，最佳帧仍为变形帧：%s" % judge.get("形变")
    out["最佳帧文件"] = _rel(tid, frame)
    wants = [w.get("要表现什么") or "商品正面全貌"
             for w in (_norm_items(judge.get("白底图"), "要表现什么") or [{}])
             ][:MAX_WHITE_BG] or ["商品正面全貌"]
    desc = judge.get("商品") or rec["product"].get("name") or "该商品"

    def one(job):
        n, want = job
        prompt = WHITE_BG_PROMPT % (desc, want)
        # 生成模型可能编造结构，白底图必须和源帧对一遍：一轮 3 张挑最好的，比对不过再来一轮。
        # 只有明确不一致才弃用——比对本身失败时保留（unknown_ok），源帧质量本就一般，
        # 这里判过严会把整条补齐链路清空
        got = _refine_rounds(prompt, frame,
                             lambda r, k: _p(tid, "product", "whitebg_%02d_r%d_%d.jpg" % (n, r, k)),
                             unknown_ok=True)
        last = (got["轮次"] or [{}])[-1]
        d = {"文件": got["最终"] or last.get("最佳"), "要表现什么": want, "prompt": prompt,
             "源帧": _rel(tid, frame), "轮次": got["轮次"],
             "原图对比": last.get("对比") or {}}
        if not d["文件"]:
            log(rec, "白底图生成失败（%s）：%d 轮都没有生成成功的候选" % (want, len(got["轮次"])))
            return {}
        if not got["通过"]:
            log(rec, "白底图与源帧商品不一致，弃用（%s）：%s"
                % (want, ((last.get("对比") or {}).get("差异") or "")[:80]))
            d["弃用"] = True
        return d

    with cf.ThreadPoolExecutor(min(3, len(wants))) as ex:
        details = [d for d in ex.map(one, list(enumerate(wants, 1))) if d]
    kept = [d for d in details if not d.get("弃用")]
    out["白底图"] = [d["文件"] for d in kept]                   # 下游按路径列表消费，口径不变
    out["白底图明细"] = [dict(d, 文件=_rel(tid, d["文件"])) for d in details]   # 报告溯源用
    log(rec, "商品图补齐：抽帧 %d 张（最佳帧 %s，%s%s），白底图 %d 张"
        % (len(out["抽帧"]), os.path.basename(frame),
           "清晰" if judge.get("清晰") else (judge.get("问题") or "不够清晰"),
           "，商品变形" if judge.get("商品变形") else "", len(out["白底图"])))
    return out


def _pick_refined(original: str, cands: list) -> dict:
    """一轮候选里挑最好的一张，并判断商品本身有没有被改掉。

    返回 {"最佳": path|None, "商品一致": True/False/None, "差异": str, "理由": str}；
    商品一致 = None 表示挑选/比对这步本身失败，保守口径交给调用方定。
    """
    if not cands:
        return {"最佳": None, "商品一致": None, "差异": "本轮没有生成成功的候选", "理由": ""}
    try:
        got = write_script._parse_json(aigc.understand(
            PICK_REFINED_PROMPT.replace("__N__", str(len(cands))),
            media=[{"type": "image", "url": u} for u in [original] + cands],
            max_tokens=512, json_mode=True))
        try:
            k = min(max(int(got.get("最佳")), 1), len(cands))
        except (TypeError, ValueError):
            k = 1
        return {"最佳": cands[k - 1], "商品一致": rules.as_bool(got.get("商品一致")),
                "差异": str(got.get("差异") or ""), "理由": str(got.get("理由") or "")}
    except Exception as exc:  # noqa: BLE001
        return {"最佳": cands[0], "商品一致": None,
                "差异": "挑选比对失败：%s" % str(exc)[:120], "理由": ""}


def _pick_state(original: str, cands: list, want: str, manifest: dict = None) -> dict:
    """中间状态图的挑选：既要还是同一个商品，又要真的把目标状态画出来了。

    manifest 是部件到位清单（装配/拆解序列才有）：给了就按「应有都在、应无都不在」判，
    不再按「和原图一模一样」判——半成品图在后一种口径下永远过不了。
    返回结构与 _pick_refined 一致（商品一致 兼表「这张能用」），差异里带上没达成的原因。
    """
    if not cands:
        return {"最佳": None, "商品一致": None, "差异": "本轮没有生成成功的候选", "理由": ""}
    blk = _manifest_block(manifest)
    try:
        got = write_script._parse_json(aigc.understand(
            PICK_STATE_PROMPT.replace("__N__", str(len(cands))).replace("__WANT__", want)
                             .replace("__MANIFEST__", blk)
                             .replace("__MANIFEST_RULE__", MANIFEST_RULE if blk else ""),
            media=[{"type": "image", "url": u} for u in [original] + cands],
            max_tokens=512, json_mode=True))
        try:
            k = min(max(int(got.get("最佳")), 1), len(cands))
        except (TypeError, ValueError):
            k = 1
        same = rules.as_bool(got.get("商品一致"))
        done = rules.as_bool(got.get("状态达成"))
        diff = str(got.get("差异") or "")
        if same is True and done is not True:
            diff = ("目标状态没做到：%s" % (diff or want)) if diff or want else "目标状态没做到"
        return {"最佳": cands[k - 1],
                "商品一致": None if (same is None or done is None) else (same and done),
                "差异": diff, "理由": str(got.get("理由") or "")}
    except Exception as exc:  # noqa: BLE001
        return {"最佳": cands[0], "商品一致": None,
                "差异": "挑选比对失败：%s" % str(exc)[:120], "理由": ""}


def _refine_rounds(prompt: str, ref: str, dst_of, unknown_ok: bool = False,
                   want: str = "", manifest: dict = None) -> dict:
    """按同一条编辑指令改造参考图，直到改出与原图同一个商品的图。

    一轮 = 生成 REFINE_BATCH 张候选（默认 1 张，一张一张跑）→ VLM 与原图比对；不过关就把
    差异点追加进指令再来一轮，最多 REFINE_ROUNDS 轮。轮次用完仍不过关返回 通过=False，
    调用方退回原图。
    unknown_ok=True 时比对本身失败也算过关（白底图的源帧质量本就一般，判过严会把补齐链路清空）。
    want 非空表示这是「中间状态图」：除了商品一致，还要判目标状态有没有真的画出来。
    manifest 非空表示这张图带部件到位清单（装配/拆解序列），判定口径整体让位给清单。
    dst_of(轮次, 序号) 给出候选图落盘路径。
    """
    out: dict = {"通过": False, "最终": None, "轮次": []}
    cur = prompt
    for r in range(1, REFINE_ROUNDS + 1):
        def one(k):
            dst = dst_of(r, k)
            try:
                url = aigc.gen_image(cur, ref_images=[ref])
                return {"文件": storage.download(url, dst), "url": url}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)[:120]}
        with cf.ThreadPoolExecutor(REFINE_BATCH) as ex:
            cands = list(ex.map(one, range(1, REFINE_BATCH + 1)))
        ok = [c["文件"] for c in cands if c.get("文件")]
        pick = _pick_state(ref, ok, want, manifest) if want else _pick_refined(ref, ok)
        out["轮次"].append({"轮次": r, "prompt": cur, "候选": cands, "最佳": pick["最佳"],
                            "对比": {k: pick[k] for k in ("商品一致", "差异", "理由")}})
        same = pick["商品一致"]
        if pick["最佳"] and (same is True or (unknown_ok and same is None)):
            out.update({"通过": True, "最终": pick["最佳"]})
            return out
        cur = prompt + "特别注意：上一轮改造出现了这些走样，必须避免：%s。" % (pick["差异"] or "商品外观走样")
    return out


def _index_product_images(rec: dict, images: list) -> list:
    """商家给的每张图做一次结构化理解，产物落 product/image_index.json。

    理解失败的图不丢：标 error 进索引，选图时自然排在后面（结构信息缺失就没法论证它更好）。
    """
    tid = rec["task_id"]
    keys = ("caption", "商品", "展示信息", "可见状态", "形变", "视角或状态", "是否局部细节",
            "完整性", "完整展示的角度", "商品占比", "同图商品数", "干扰", "画质")

    def one(job):
        n, path = job
        d = {"编号": n, "文件": path}
        try:
            got = write_script._parse_json(aigc.understand(
                IMAGE_INDEX_PROMPT, media=[{"type": "image", "url": path}],
                max_tokens=1024, json_mode=True))
        except Exception as exc:  # noqa: BLE001
            d.update({"caption": "结构化理解失败：%s" % str(exc)[:120], "error": True})
            return d
        d.update({k: got.get(k) for k in keys})
        return d

    with cf.ThreadPoolExecutor(min(4, len(images))) as ex:
        index = list(ex.map(one, enumerate(images, 1)))
    path = _p(tid, "product", "image_index.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([dict(d, 文件=d["文件"]) for d in index], fh, ensure_ascii=False, indent=2)
    whole = sum(1 for d in index if str(d.get("完整性") or "").startswith(("完整", "基本完整")))
    part = sum(1 for d in index if rules.as_bool(d.get("是否局部细节")) is True)
    dirty = sum(1 for d in index if d.get("干扰"))
    bent = sum(1 for d in index if _anchor_block(d).startswith("商品变形"))
    log(rec, "商家图片结构化理解：%d 张（完整/基本完整 %d，局部特写 %d，带干扰 %d，商品变形 %d）"
        % (len(index), whole, part, dirty, bent))
    return index


def _anchor_block(d: dict) -> str:
    """这张图不能当 @图片1（主锚点）的理由；能当就返回空串。

    两条都是硬口径，判不出来一律放行——免得把本该当主锚点的好图挤下去：
    - 多款/多配色同框：主锚点自带多种配色等于要求生成端混搭（实测四色全览图当 @图片1，
      场景设定图直接生成了 4 台不同颜色的手机）。体检本来会把多余那几台删掉，
      但轮次用完保留原图时多台还在，所以选图这里再兜一层。
    - 商品被人为挤压/按压/揉搓/拉伸而变形：主锚点是生成端照着画的形状基准，
      拿变形的当基准，成片里的商品会一路变形下去（耳塞被搓成细长条那种图最典型）。
    提示词里已经写了同样两条口径，这里是确定性兜底：模型不照做时代码仍然换位。
    """
    try:
        if int(d.get("同图商品数")) >= 2:
            return "%d 台同框" % int(d["同图商品数"])
    except (TypeError, ValueError):
        pass
    if _is_deformed(d.get("形变")):
        return "商品变形（%s）" % str(d.get("形变")).strip()[:20]
    return ""


def _select_refs(rec: dict, index: list) -> tuple:
    """按结构化理解挑参考图并排序，返回（重排后的图列表, 选图明细）。

    只读 image_index，不再看图：选图是「覆盖哪些信息」的取舍，结构化理解已经把这些
    信息摊平了。落选的图仍留在列表尾部（事实卡可查、不静默丢），只是排不进 @图片N。
    第 1 张必须是完整全貌，且不能多款同框、不能是变形的商品——生成端拿第 1 张当主锚点，
    硬口径见 _anchor_block。
    """
    by_no = {d["编号"]: d for d in index}
    brief = [{k: v for k, v in d.items() if k != "文件" and v not in (None, "", [])}
             for d in index]
    try:
        got = write_script._parse_json(aigc.understand(
            SELECT_REFS_PROMPT.replace("__NAME__", rec["product"].get("name") or "该商品")
                              .replace("__INDEX__", json.dumps(brief, ensure_ascii=False, indent=1))
                              .replace("__MAX__", str(rules.PRODUCT_REF_MAX)),
            max_tokens=2048, json_mode=True))
    except Exception as exc:  # noqa: BLE001
        log(rec, "参考图选图失败，按原顺序用：%s" % str(exc)[:120])
        return [d["文件"] for d in index], {"说明": "选图失败，按原顺序取前几张当参考图"}

    # chosen 与 picked 同步收集：不能事后拿 picked 的长度去切原始「选中」列表——
    # 模型给了不存在或重复的编号时，非法条目会留在切片里、真正入选的那张反被切掉，
    # 而下游 product_facts 要靠这份明细里的「角色」保护多配色图不被删款。
    picked, seen, chosen = [], set(), []
    for c in _norm_items(got.get("选中"), "编号"):
        d = by_no.get(_as_no(c.get("编号"), by_no))
        if not d or d["文件"] in seen:
            continue
        seen.add(d["文件"])
        c["文件"] = d["文件"]
        c["caption"] = d.get("caption")
        picked.append(d["文件"])
        chosen.append(c)
    if not picked:
        log(rec, "选图没给出有效编号，按原顺序用")
        return [d["文件"] for d in index], {"说明": "选图结果无效，按原顺序取前几张当参考图"}
    picked = picked[:rules.PRODUCT_REF_MAX]
    # 主锚点硬口径兜底：@图片1 不能多款同框、也不能是变形的商品，理由见 _anchor_block
    by_file = {d["文件"]: d for d in index}
    swap = ""
    blocked = _anchor_block(by_file.get(picked[0]) or {})
    if len(picked) > 1 and blocked:
        alt = next((p for p in picked[1:] if not _anchor_block(by_file.get(p) or {})), "")
        if alt:
            swap = "%s（%s）让位给 %s" % (os.path.basename(picked[0]), blocked,
                                          os.path.basename(alt))
            picked.remove(alt)
            picked.insert(0, alt)
    rest = [d["文件"] for d in index if d["文件"] not in set(picked)]
    dropped = _norm_items(got.get("落选"), "编号")
    for c in dropped:
        c["文件"] = (by_no.get(_as_no(c.get("编号"), by_no)) or {}).get("文件")
    chosen = chosen[:len(picked)]
    # 换过位就把明细也按新顺序排：选图明细的第 1 条要始终对应 @图片1
    order = {f: i for i, f in enumerate(picked)}
    chosen.sort(key=lambda c: order.get(c.get("文件"), len(order)))
    sel = {"选中": chosen, "落选": dropped, "缺失": got.get("缺失") or []}
    if swap:
        sel["主锚点换位"] = swap
    log(rec, "参考图选图：%d 张里选 %d 张，@图片1 = %s（%s）%s%s"
        % (len(index), len(picked), os.path.basename(picked[0]),
           (sel["选中"][0].get("角色") or "") if sel["选中"] else "",
           "，缺失：%s" % "、".join(str(x) for x in sel["缺失"]) if sel["缺失"] else "",
           "，主锚点换位：%s" % swap if swap else ""))
    return picked + rest, sel


def _refine_one(rec: dict, path: str, item: dict, dst_of, role: str = "") -> dict:
    """一张商品图的编辑判断 + 改造，返回体检明细（原图 / 最终 / 判定 / 结论 / 轮次）。

    结构化理解里没有干扰、商品占比不小的图直接用，不再花一次 VLM 调用。
    需要编辑的两步：VLM 看图 + 已有结构化理解，一次写好点名到具体文字/贴片的编辑指令
    （EDIT_PROMPT_PROMPT）→ 按这条指令走 _refine_rounds 多轮改造直到与原图比对通过；
    轮次用完仍不通过就保留原图。判定/生成失败一律保留原图——体检只能让参考图变好，
    不能把图弄丢。dst_of(轮次, 序号) 给出候选图落盘路径。
    """
    d = {"原图": path, "最终": path}
    dirty = [str(x) for x in (item.get("干扰") or []) if str(x).strip()]
    if item and not item.get("error") and not dirty and item.get("商品占比") != "小":
        d.update({"判定": {"商品": item.get("商品"), "可直接用": True, "问题": "",
                          "依据": "结构化理解：无干扰、商品占比%s" % (item.get("商品占比") or "?")},
                  "结论": "结构化理解判定干净，直接使用（未再调用 VLM）"})
        return d
    hint = ""
    if item and not item.get("error"):
        hint = ("已有的结构化理解（可直接采信）：%s\n"
                % json.dumps({k: item.get(k) for k in ("caption", "干扰", "商品占比",
                                                       "同图商品数", "视角或状态")},
                             ensure_ascii=False))
    if role:      # 用途决定多款同框怎么处理：展示多配色的图不许删款，其余只留一款
        hint += "这张图被选中的用途：%s\n" % role
    try:
        verdict = write_script._parse_json(aigc.understand(
            EDIT_PROMPT_PROMPT.replace("__HINT__", hint),
            media=[{"type": "image", "url": path}], max_tokens=1024, json_mode=True))
    except Exception as exc:  # noqa: BLE001
        d.update({"判定": {"可直接用": True,
                           "问题": "判定失败，按可用处理：%s" % str(exc)[:120]},
                  "结论": "判定失败，保留原图"})
        return d
    d["判定"] = {k: verdict.get(k) for k in ("商品", "可直接用", "问题")}
    if rules.as_bool(verdict.get("可直接用")) is not False:
        d["结论"] = "直接使用"
        return d
    prompt = str(verdict.get("改写prompt") or "").strip()
    if not prompt:       # 判了不可用却没给指令：按通用口径兜底，别让这张图漏过体检
        name = verdict.get("商品") or rec["product"].get("name") or "商品"
        prompt = ("「%s」保持不变，保持包装上印刷的文字不变，删除画面上叠加的所有营销文字、"
                  "促销贴片与水印。修改为白底图，「%s」占据主要画面。"
                  "生成商品主图，「%s」居于正中，放大。" % (name, name, name))
        d["指令兜底"] = True
    d["prompt"] = prompt
    got = _refine_rounds(prompt, path, dst_of)
    d["轮次"] = got["轮次"]
    last = (got["轮次"] or [{}])[-1].get("对比") or {}
    d["对比"] = last
    if got["通过"]:
        d.update({"最终": got["最终"],
                  "结论": "已改造（第%d轮，与原图比对通过）" % len(got["轮次"])})
    else:
        d["结论"] = ("%d 轮改造都与原图不一致，保留原图（%s）"
                     % (len(got["轮次"]), (last.get("差异") or "比对失败")))
    return d


def _refine_user_images(rec: dict, images: list, index: list = None,
                        limit: int = None, roles: dict = None) -> tuple:
    """选中的商品图逐张体检，返回（替换后的图列表, 检查明细）。

    只体检会进生成参考的那几张（limit，默认前 PRODUCT_REF_MAX 张）；剩下的图分镜级选图
    才可能用到，被选中时由 _plan_shot_refs 补体检，避免在这里为几十张图白花 VLM 调用。
    """
    tid = rec["task_id"]
    idx_by_path = {d["文件"]: d for d in (index or [])}

    def one(job):
        n, path = job
        return _refine_one(rec, path, idx_by_path.get(path) or {},
                           lambda r, k: _p(tid, "product",
                                           "refined_%02d_r%d_%d.jpg" % (n, r, k)),
                           role=(roles or {}).get(path, ""))

    n_head = min(limit or rules.PRODUCT_REF_MAX, rules.PRODUCT_REF_MAX)
    head = list(enumerate(images[:n_head], 1))
    with cf.ThreadPoolExecutor(min(3, len(head))) as ex:
        details = list(ex.map(one, head))
    final = [d["最终"] for d in details] + list(images[n_head:])
    need = sum(1 for d in details
               if rules.as_bool((d.get("判定") or {}).get("可直接用")) is False)
    fixed = sum(1 for d in details if d["最终"] != d["原图"])
    log(rec, "参考图编辑判断：%d 张，判需编辑 %d 张，改造替换 %d 张（其余保留原图）"
        % (len(details), need, fixed))
    return final, details


# 参考图需求是剧本级的：卡点换装 / 多配色对比这类剧本，每一拍要出现的是不同的商品状态，
# 全片共用同一组商品图会让模型把几款混搭成四不像。所以剧本出来后再按镜分配参考图。
SHOT_REFS_PROMPT = """给每一个要 AI 补片的分镜挑商品参考图。商品：__NAME__。

【可用商品图】编号 + 这张图的结构化理解
__IMAGES__

【要补片的分镜】
__SHOTS__

挑图口径：
- 一镜最多 __MAX__ 张，够用就少给：只给这一镜画面里真正要出现的角度 / 状态 / 配色；
  信息重复的图（同角度同款）只带一张
- 换装、换色、多款对比这种每拍换一个对象的卡点，每镜只给它自己那一张，别把同类图都塞进来
- 画面里不出现商品（纯人物镜、纯氛围镜、纯文字镜）就给空数组
- 拿不准就给最能代表商品整体的那一张——参考图越多，模型越容易混搭出四不像
- 只能用上面列出的编号

再判断这一镜要不要「中间状态图」：视频模型一次画不准的高精度、强结构约束、易出错的视觉状态，
先用图像编辑把这个状态锁成一张图，再让视频模型只负责状态之间的动态变化。
需要的典型情形：商品开合/展开/拆装、屏幕或指示灯的显示内容、画面要出现大量可读文字或
参数界面（AI 直接画文字必乱码，先锁成状态图）、贴合到身体或器物上的状态、
多款商品同框的固定摆位、必须正确的图案或印刷内容、换装换色的目标形态。
只是镜头运动、光影变化、人物表演、环境氛围这些，视频模型自己能处理，不要提。
判要的时候先在上面的可用图里找：已经就是这个状态的图直接用（填「状态来源编号」），
找不到才写编辑指令去生成（填「状态图基于编号」+「状态图编辑指令」，按「「商品」保持不变，…」的句式写，
只能基于该图里看得见的角度改状态，不许换角度、不许凭空补看不到的结构）。
状态忠实硬口径：状态只能取可用图里真实看得见的形态，或黑屏/亮屏这类中性形态。
具体的显示内容、印刷图案、成段文字这类必须照实抄的状态，不许凭空写编辑指令去造——
那等于编一个假商品。
但你只拿到上面这份文字理解、看不到原图，「文字里没提到」不等于「素材里没有」，
所以这种状态不要在这里下结论：把可能已经是这个状态的图都填进「状态候选编号」（拿不准就多填几张），
后面会真的把这些图打开核验，命中就直接拿来用。
只有你能确定所有可用图都不可能有（例如商品根本没有屏幕），才写进「缺失」，说明里写清为什么不可能。

最后判断整支剧本里有没有「装配序列」：一连串镜头拍的是同一件商品逐步成形或逐步拆开，
每一镜画面里的商品都是故意不完整的（经典例子：从一块纯白无部件的机身白板开始，
镜头一个个把中框、屏幕、摄像头装上去，最后一镜才是完整成品）。
这类镜头的中间图必须允许与原图不一致，所以要额外给出每一镜的「部件到位清单」。

判定装配序列的门槛要高，宁可不给：必须是分镜里明写了这种逐步成形/逐步拆解的叙事，
且这些镜头在时间上连续。只是换角度、换配色、开合一次、点亮屏幕，都不算装配序列，
按上面普通「关键状态」处理。剧本里没有这种叙事就给空数组——
清单是「应无」的唯一来源，随便给等于把商品一致性校验关掉了。

序列内部的硬要求：
- 「基准编号」整条序列只能有一个，且必须是完整商品的那张图：所有中间图都从它减部件生成，
  中途换图会让商品角度跳变，整段废掉。挑角度最完整、能看见最多部件的那张
- 「应有」「应无」都只能写这张基准图里看得见的部件，看不见的（内部结构、背面）不许写
- 相邻两镜的部件集合必须嵌套：装配就一路只增不减，拆解就一路只减不增，不许来回变
- 「应有」不能为空（至少要有机身轮廓这类主体），装配序列的最后一镜「应无」应为空（=完整成品）
- 每一镜给一条「编辑指令」，从基准图减部件的写法：
  「手机」保持不变，删除后置摄像头模组、删除闪光灯，删除处呈平整同材质表面。

输出 json：
{"分镜": [{"镜头": 3, "参考图编号": [4], "理由": "这一镜拍紫色机背面，编号4是背面全貌",
           "关键状态": "不需要就写空字符串，如 副屏点亮显示时钟界面",
           "状态来源编号": 已有图就是这个状态时填编号，否则 null,
           "状态候选编号": [看文字理解拿不准、但可能已经是这个状态的图编号，交给看图核验；没有就空数组],
           "状态图基于编号": 需要生成时基于哪张图改，否则 null,
           "状态图编辑指令": "需要生成时的图像编辑指令，否则空字符串"}],
 "装配序列": [{"镜头": [3, 4, 5], "基准编号": 2, "叙事类型": "装配或拆解",
              "叙事": "从白板逐个装上零件到成品",
              "每镜": [{"镜头": 3, "应有": ["机身白板轮廓"],
                       "应无": ["屏幕", "后置摄像头模组", "机身印刷文字"],
                       "编辑指令": "从基准图减部件的图像编辑指令"}]}],
 "缺失": [{"镜头": 5, "需要": "正面全貌", "说明": "可用图里没有这个角度"}]}

每一镜都要出现在「分镜」里。只输出 json。"""

SHOT_BRIEF_FIELDS = ("序号", "时长秒", "景别", "运镜", "画面", "动作", "叙事功能")


def _as_no(v, by_no: dict):
    """把模型给的商品图编号归一成 by_no 里的 int 编号；给不出就返回 None。"""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n in by_no else None


def _shot_int(v):
    """镜号归一成 int；缺失或非数字返回 None（调用方按「这一镜不做计划」处理）。

    剧本模型漏写某一镜的 序号 时 plan_segments 会把 None 原样写进 分段[].镜头序号
    （上游是刻意容忍的：write_script._shot_no、shot_match._no_key 都吞掉了这种脏值），
    所以这里既不能 int(None) 直接崩，也不能只归一一边——两边类型不一致会让筛选静默落空。
    """
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _norm_assembly(seqs: list, want: set, by_no: dict, idx: dict) -> tuple:
    """校验并规整装配序列，返回（镜号str -> 部件到位清单, 丢弃原因列表）。

    清单是「应无非空」的唯一来源：不在序列里的镜头永远按强一致口径判，模型没法靠往「应无」
    里塞部件来绕开商品一致性校验。所以这里的校验只做结构性检查、一条不过就整条丢掉——
    宁可少一层创意，也不能把一致性门禁放开。
    结构性约束不信模型自己说的：基准图必须钉死一张（中途换图商品角度会跳变，整段废）、
    相邻两镜的部件集合必须嵌套（装配一路只增，拆解一路只减）。
    """
    out, drops = {}, []
    for seq in seqs or []:
        try:
            base = int(seq.get("基准编号"))
        except (TypeError, ValueError):
            base = None
        if base not in by_no:
            drops.append("基准编号 %s 不在可用商品图里" % seq.get("基准编号"))
            continue
        steps = []
        for s in seq.get("每镜") or []:
            try:
                no = int(s.get("镜头"))
            except (TypeError, ValueError):
                continue
            if no not in want or any(no == x[0] for x in steps):
                continue
            have = [str(x).strip() for x in (s.get("应有") or []) if str(x).strip()]
            none = [str(x).strip() for x in (s.get("应无") or []) if str(x).strip()]
            steps.append((no, have, none, str(s.get("编辑指令") or "").strip()))
        steps.sort(key=lambda x: x[0])
        if len(steps) < 2:
            drops.append("序列 %s 有效镜头不足 2 个" % (seq.get("镜头") or ""))
            continue
        if any(not have for _, have, _, _ in steps):
            drops.append("镜%s 的「应有」为空" % "、".join(
                str(no) for no, have, _, _ in steps if not have))
            continue
        if not any(none for _, _, none, _ in steps):
            drops.append("序列 %s 每一镜都是完整商品，不需要中间图"
                         % "、".join(str(no) for no, _, _, _ in steps))
            continue
        # 相邻两镜的应有集合必须嵌套，且方向全程一致：装配只增，拆解只减
        sets = [set(have) for _, have, _, _ in steps]
        grow = all(a <= b for a, b in zip(sets, sets[1:]))
        shrink = all(b <= a for a, b in zip(sets, sets[1:]))
        if not (grow or shrink):
            drops.append("序列 %s 的部件集合来回变，不是单调的装配/拆解"
                         % "、".join(str(no) for no, _, _, _ in steps))
            continue
        narr = "装配" if grow else "拆解"
        total = len(steps)
        for i, (no, have, none, edit) in enumerate(steps, 1):
            out[str(no)] = {
                "叙事类型": narr, "序号": i, "共几步": total, "基准编号": base,
                "应有": have, "应无": none, "叙事": str(seq.get("叙事") or ""),
                "编辑指令": edit or ("「%s」保持不变，删除%s，删除处呈平整同材质表面。"
                                    % ((idx.get(base) or {}).get("商品") or "商品",
                                       "、删除".join(none))),
            }
    return out, drops


def _apply_assembly(rec: dict, plan: dict, seqs: list, want: set, by_no: dict,
                    idx: dict) -> None:
    """把装配序列的部件到位清单合进分镜计划（原地改 plan）。

    这些镜头的中间图是故意不完整的，判定口径整体交给部件到位清单；结构性校验不过就整条丢掉，
    这些镜头退回普通选图（= 强一致口径）。
    """
    got, drops = _norm_assembly(seqs or [], want, by_no, idx)
    for no, m in got.items():
        v = plan.get(no)
        if not v:
            continue
        # 应无为空的那一步就是完整商品，基准图本身即答案：直接复用，别再生成一张近似的
        v.update({"装配清单": m,
                  "状态来源编号": m["基准编号"] if not m["应无"] else None,
                  "状态图基于编号": m["基准编号"],
                  "状态图编辑指令": m["编辑指令"],
                  "关键状态": "%s第 %d/%d 步：有 %s；无 %s"
                              % (m["叙事类型"], m["序号"], m["共几步"],
                                 "、".join(m["应有"]), "、".join(m["应无"]) or "（无，完整商品）")})
    if got or drops:
        log(rec, "装配序列：%s%s"
            % ("%d 镜带部件清单（镜%s）" % (len(got), "、".join(sorted(got, key=int)))
               if got else "无",
               "，丢弃 %d 条（%s）" % (len(drops), "；".join(drops)[:200]) if drops else ""))


def _pin_base_files(plan: dict, by_no: dict) -> None:
    """把中间状态图的基准图钉死到文件路径（原地改 plan）。

    体检可能把 by_no 里的文件换成改造版，而 fact_card.json 不回写，断点续跑时
    _make_state_images 从 fact_card 重建的 detail 会退回未体检原图，基准图就换人了
    （实测 72 张商品图里 23 张会被体检替换）。装配序列尤其不能容忍——整条序列必须同一张基准图。
    """
    for v in plan.values():
        for key, field in (("状态图基于编号", "状态图基准文件"),
                           ("状态来源编号", "状态图来源文件")):
            d = by_no.get(v.get(key))
            if d and d.get("文件"):
                v[field] = d["文件"]


def _vet_picked(rec: dict, plan: dict, by_no: dict, idx: dict) -> list:
    """分镜级选中、但事实卡那一轮没体检过的图，补做一次编辑判断/改造。

    事实卡只体检了当全局参考图的前 PRODUCT_REF_MAX 张；换装/多配色素材几十张，分镜级选图
    会取到后面那些没体检的原始实拍图——常带真人脸与营销贴片，直接下发会被视频模型的真人
    风控拒掉，整段掉到 t2v，商品信息全丢。所以「被选中即体检」。
    改造后原来的 url 失效，置空让 _url_of 重新上传。
    """
    nos = []
    for v in plan.values():
        for n in (list(v.get("参考图编号") or [])
                  + [v.get("状态来源编号"), v.get("状态图基于编号")]):
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if n not in nos and by_no.get(n) and not by_no[n].get("已体检"):
                nos.append(n)
    if not nos:
        return []
    tid = rec["task_id"]

    roles = {}
    for v in plan.values():
        for n in (v.get("参考图编号") or []) + [v.get("状态来源编号"), v.get("状态图基于编号")]:
            try:
                roles.setdefault(int(n), str(v.get("理由") or ""))
            except (TypeError, ValueError):
                pass

    def one(n):
        d = by_no[n]
        return n, _refine_one(rec, d["文件"], idx.get(n) or {},
                              lambda r, k: _p(tid, "product",
                                              "refined_s%s_r%d_%d.jpg" % (n, r, k)),
                              role=roles.get(n, ""))

    with cf.ThreadPoolExecutor(min(3, len(nos))) as ex:
        out = list(ex.map(one, nos))
    for n, got in out:
        d = by_no[n]
        d["已体检"] = True
        d["体检"] = got
        if got["最终"] != got["原图"]:
            d.update({"文件": got["最终"], "url": None})
    fixed = sum(1 for _, g in out if g["最终"] != g["原图"])
    log(rec, "分镜级选图补体检：%d 张（编号%s），改造替换 %d 张"
        % (len(out), "、".join(str(n) for n, _ in out), fixed))
    return [dict(g, 编号=n) for n, g in out]


def _plan_shot_refs(rec: dict, built: dict, product: dict, shot_nos: list) -> dict:
    """按剧本给每一镜分配商品参考图，产物 generated/shot_refs.json。

    返回 {"分镜": {镜号: {"参考图编号","urls","理由"}}, "缺失": [...]}；
    只给要 AI 补片的镜头做计划（裁素材的镜头本来就不下发参考图）。
    没有商品图明细（素材抽帧补齐的任务、老任务）或调用失败都返回空，
    调用方退回「全片共用前几张」的老口径，不阻断生成。
    """
    tid = rec["task_id"]
    path = _p(tid, "generated", "shot_refs.json")
    if os.path.isfile(path):          # 断点续跑复用，别再问模型一次
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    # 候选池是全部商品图，不止事实卡里当兜底的前 PRODUCT_REF_MAX 张：多款/多配色的素材
    # （换装卡点常见几十张）必须每镜都能取到自己那一款，只是单次请求最多带 PRODUCT_REF_MAX 张。
    # 事实卡只预上传了前几张，其余等被选中时再上传（省掉几十次无用上传）。
    detail = [d for d in (product.get("商品图明细") or [])
              if d.get("文件") and d.get("编号") is not None]
    if not detail or not shot_nos:
        return {}

    def _url_of(d: dict) -> str:
        if not d.get("url"):
            d["url"] = storage.upload(d["文件"])
        return d["url"]

    idx = {d.get("编号"): d for d in (product.get("商品图索引") or [])}
    imgs = []
    for d in detail:
        item = idx.get(d["编号"]) or {}
        imgs.append({"编号": d["编号"],
                     "caption": item.get("caption") or os.path.basename(d["文件"]),
                     "视角或状态": item.get("视角或状态"), "完整性": item.get("完整性"),
                     "展示信息": item.get("展示信息"), "可见状态": item.get("可见状态"),
                     "已改造为白底图": d["文件"] != d["原图"]})
    want = {n for n in (_shot_int(x) for x in shot_nos) if n is not None}
    if not want:
        return {}
    brief = [{k: s.get(k) for k in SHOT_BRIEF_FIELDS if s.get(k) not in (None, "", [])}
             for s in (built["剧本"].get("分镜") or []) if _shot_int(s.get("序号")) in want]
    by_no = {d["编号"]: d for d in detail}
    try:
        got = write_script._parse_json(aigc.understand(
            SHOT_REFS_PROMPT.replace("__NAME__", product.get("name") or "该商品")
                            .replace("__IMAGES__", json.dumps(imgs, ensure_ascii=False, indent=1))
                            .replace("__SHOTS__", json.dumps(brief, ensure_ascii=False, indent=1))
                            .replace("__MAX__", str(rules.PRODUCT_REF_MAX)),
            max_tokens=4096, json_mode=True))
    except Exception as exc:  # noqa: BLE001
        log(rec, "分镜参考图分配失败，全片共用前几张商品图：%s" % str(exc)[:120])
        return {}
    plan = {}
    for r in got.get("分镜") or []:
        try:
            no = int(r.get("镜头"))
        except (TypeError, ValueError):
            continue
        if no not in want:
            continue
        nos = []
        for n in (r.get("参考图编号") or [])[:rules.PRODUCT_REF_MAX]:
            try:
                d = by_no.get(int(n))
            except (TypeError, ValueError):
                d = None
            if d and d["编号"] not in nos:
                nos.append(d["编号"])
        plan[str(no)] = {"参考图编号": nos, "urls": [], "理由": r.get("理由") or "",
                         "关键状态": str(r.get("关键状态") or "").strip(),
                         # 这两个编号要拿去查 detail（键是 int），模型写成 "3" 时不归一就查不中，
                         # 中间状态图会静默退回普通商品图。参考图编号上面已经 int 过了，口径要一致
                         "状态来源编号": _as_no(r.get("状态来源编号"), by_no),
                         # 文字理解里判不出状态在不在，就靠这批候选交给 _verify_pending_states 看图裁定
                         "状态候选编号": [n for n in
                                          (_as_no(x, by_no) for x in (r.get("状态候选编号") or []))
                                          if n is not None],
                         "状态图基于编号": _as_no(r.get("状态图基于编号"), by_no),
                         "状态图编辑指令": str(r.get("状态图编辑指令") or "").strip()}
    if not plan:
        log(rec, "分镜参考图分配没给出有效编号，全片共用前几张商品图")
        return {}
    _apply_assembly(rec, plan, got.get("装配序列"), want, by_no, idx)
    # 选中即体检：改造后 by_no 里的文件已换，urls 统一在体检之后才上传
    vet = _vet_picked(rec, plan, by_no, idx)
    for v in plan.values():
        v["urls"] = [_url_of(by_no[n]) for n in v["参考图编号"]]
    _pin_base_files(plan, by_no)
    out = {"分镜": plan, "缺失": got.get("缺失") or [], "可用商品图": imgs}
    if vet:
        out["选中补体检"] = vet
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    miss = len(want) - len(plan)
    states = [no for no, v in plan.items() if v["关键状态"]]
    log(rec, "分镜参考图分配：%d 镜（每镜 %.1f 张%s）%s%s"
        % (len(plan), sum(len(v["urls"]) for v in plan.values()) / max(1, len(plan)),
           "，%d 镜未覆盖按全局图兜底" % miss if miss > 0 else "",
           "，剧本要的角度缺 %d 项" % len(out["缺失"]) if out["缺失"] else "",
           "，%d 镜要中间状态图（镜%s）" % (len(states), "、".join(states)) if states else ""))
    return out


def _make_state_images(rec: dict, shot_refs: dict, product: dict) -> dict:
    """把分镜计划里要的「中间状态图」备好：先用已有图匹配，没有才走图像编辑生成。

    锁死状态再交给视频模型，是因为视频模型一次性完成高精度状态变化的成功率很低
    （开合、界面显示、指示灯这类一画就错）。生成走 _refine_rounds（一张一张跑，每张都验），
    验收多一条「状态达成」；两轮都不过就放弃这张图，本镜退回普通商品图，只是少一层保障。
    带「装配清单」的镜头是装配/拆解序列的一步：全部从同一张基准图减部件生成，彼此不依赖，
    所以整条序列可以并发出图——不能链式基于上一张改，那样误差会逐张累积，也说不清走样是
    哪一轮引入的。产物写回 generated/shot_refs.json，断点续跑直接复用。
    """
    plan = (shot_refs or {}).get("分镜") or {}
    todo = [(no, v) for no, v in plan.items()
            if v.get("关键状态") and not (v.get("中间图") or {}).get("url")]
    if not todo:
        return shot_refs
    tid = rec["task_id"]
    detail = {d.get("编号"): d for d in (product.get("商品图明细") or [])
              if d.get("编号") is not None}

    def one(job):
        no, v = job
        want = v["关键状态"]
        manifest = v.get("装配清单") or None
        # 计划里钉死的路径优先：断点续跑时 detail 是从 fact_card 重建的，体检替换过的文件不在里面
        src_file = v.get("状态图来源文件")
        src = detail.get(v.get("状态来源编号")) or {}
        src_file = src_file or src.get("文件")
        if src_file and os.path.isfile(src_file):        # 已有图就是这个状态，不用生成
            url = src.get("url") if src.get("文件") == src_file else None
            if not url:
                url = storage.upload(src_file)
                if src.get("文件") == src_file:
                    src["url"] = url
            d = {"url": url, "文件": src_file, "关键状态": want,
                 "来源": "已有商品图%s（无需生成）" % (v.get("状态来源编号") or src.get("编号"))}
            if manifest:
                d["装配清单"] = manifest
            return no, d
        base = detail.get(v.get("状态图基于编号")) or detail.get(
            (v.get("参考图编号") or [None])[0]) or {}
        base_file = v.get("状态图基准文件") or base.get("文件")
        base_no = v.get("状态图基于编号") or base.get("编号")
        prompt = v.get("状态图编辑指令")
        if not base_file or not os.path.isfile(base_file) or not prompt:
            # 没有编辑指令不等于这个状态素材里没有：挑图那步是纯文字推理，判不出来的会把候选
            # 填进「状态候选编号」。候选不够就把参考图和其余可用图都算上，交给看图那步裁定。
            cands = []
            for n in ((v.get("状态候选编号") or []) + (v.get("参考图编号") or []) + list(detail)):
                if n in detail and n not in cands:
                    cands.append(n)
            return no, {"关键状态": want, "待核验": cands[:MAX_STATE_LOOKUP],
                        "来源": "等看图核验素材里有没有这个状态"}
        got = _refine_rounds(prompt, base_file,
                             lambda r, k: _p(tid, "product", "state_%s_r%d_%d.jpg" % (no, r, k)),
                             want=want, manifest=manifest)
        d = {"关键状态": want, "prompt": prompt, "基准图": base_file,
             "基准编号": base_no, "轮次": got["轮次"]}
        if manifest:
            d["装配清单"] = manifest
        last = (got["轮次"] or [{}])[-1].get("对比") or {}
        if got["通过"]:
            d.update({"文件": got["最终"], "url": storage.upload(got["最终"]),
                      "来源": "图像编辑生成（第%d轮通过）" % len(got["轮次"])})
        else:
            d["来源"] = ("%d 轮都没做出这个状态，放弃中间图（%s）"
                        % (len(got["轮次"]), last.get("差异") or "比对失败"))
        return no, d

    # 装配序列一条就有 4~6 镜，按 3 并发要排两三轮；每张自己内部还要串行改到过关
    # （最多 REFINE_ROUNDS 轮，每轮一次生图 + 一次比对），并发放宽一点省的就是这部分等待
    with cf.ThreadPoolExecutor(min(6, len(todo))) as ex:
        for no, d in ex.map(one, todo):
            plan[no]["中间图"] = d
    _verify_pending_states(plan, detail)
    ok = [no for no, v in plan.items() if (v.get("中间图") or {}).get("url")]
    with open(_p(tid, "generated", "shot_refs.json"), "w", encoding="utf-8") as fh:
        json.dump(shot_refs, fh, ensure_ascii=False, indent=2)
    log(rec, "中间状态图：%d 镜要，备好 %d 张（%s）"
        % (len(todo), len(ok), "、".join("镜%s %s" % (no, (plan[no]["中间图"].get("来源") or ""))
                                         for no, _ in todo)))
    return shot_refs


def _find_state_image(want: str, files: list) -> dict:
    """在素材图里找「已经就是这个状态」的那张。

    files 是 [(编号, 文件)]。命中返回 {"编号", "理由"}，没命中或核验失败只返回 {"理由"}——
    核验失败按没命中处理：宁可少一层保障，也不能把一张没验过的图当成状态已锁死交给视频模型。
    """
    if not files:
        return {"理由": "没有可核验的素材图"}
    try:
        got = write_script._parse_json(aigc.understand(
            STATE_LOOKUP_PROMPT.replace("__N__", str(len(files))).replace("__WANT__", want),
            media=[{"type": "image", "url": p} for _, p in files],
            max_tokens=512, json_mode=True))
    except Exception as exc:  # noqa: BLE001
        return {"理由": "核验失败：%s" % str(exc)[:120]}
    try:
        k = int(got.get("命中"))
    except (TypeError, ValueError):
        k = 0
    out = {"理由": str(got.get("理由") or "")}
    if 1 <= k <= len(files):
        out["编号"] = files[k - 1][0]
    return out


def _verify_pending_states(plan: dict, detail: dict) -> None:
    """把文字层判不出来的关键状态，真的打开素材图核验一遍，就地写回「中间图」。

    这是整条链路上唯一有资格回答「这个状态素材里到底有没有」的地方：分镜挑参考图那步只拿到
    image_index 的文字理解、看不到原图，它说不出来的一律走到这里，命中就直接把那张素材图
    当中间状态图（和「已有商品图」路径同一个待遇），确认没有才放弃、本镜退回普通商品图。
    同一个状态 + 同一批候选图只核验一次：一支片里常有好几镜要同一个状态（时钟、开盖、亮屏）。
    """
    groups = {}
    for no, v in plan.items():
        d = v.get("中间图") or {}
        if d.get("待核验"):
            groups.setdefault((d["关键状态"], tuple(d["待核验"])), []).append(no)
    if not groups:
        return

    def one(key):
        want, cands = key
        files = [(n, detail[n]["文件"]) for n in cands
                 if (detail.get(n) or {}).get("文件") and os.path.isfile(detail[n]["文件"])]
        return key, _find_state_image(want, files)

    with cf.ThreadPoolExecutor(min(4, len(groups))) as ex:
        hits = dict(ex.map(one, list(groups)))
    for key, shots in groups.items():
        hit = hits.get(key) or {}
        src = detail.get(hit.get("编号")) or {}
        url = None
        if src.get("文件"):
            url = src.get("url") or storage.upload(src["文件"])
            src["url"] = url
        for no in shots:
            d = plan[no]["中间图"]
            d.pop("待核验", None)
            if url:
                d.update({"url": url, "文件": src["文件"],
                          "来源": "看图核验：已有商品图%s 就是这个状态" % src.get("编号")})
            else:
                d["来源"] = ("看图核验：素材里没有这个状态，放弃中间图（%s）"
                            % (hit.get("理由") or "候选图都不命中"))


def _collapse_states(hit: list) -> dict:
    """这一块的中间状态图 → {url: 状态说明}，装配序列会被收敛成起始/结束两张。

    连续的 AI 补片镜头会被 _seg_blocks 并成一块、一次生成（seedance 最短出 4s，一镜一块
    既费额度又打乱节奏），而装配序列本身就是一串连续镜头——所以一条 4~6 镜的序列通常整条
    落在同一次调用里。这时候不能把每一步的状态图都当参考图下发：
    - PIECE_REF_MAX 只有 4 张，4 步以上就会把商品原图全挤掉，模型只看得到半成品；
    - 每张都声明「必须照它画」等于给一次调用四条互斥的指令，代码层面就不自洽。
    正确的表达是「这一段从起始形态演变到结束形态」：只留首尾两张，各自标明位置。
    「块内始终应无」= 块内各步应无的交集，也就是整段视频里从头到尾都不该出现的部件——
    成片验收只能按它判，中间某一帧该不该有屏幕，整段视频的验收模型分不出来。
    """
    steps, states = [], {}
    for no, p in hit:
        mid = p.get("中间图") or {}
        if not mid.get("url"):
            continue
        m = mid.get("装配清单") or {}
        item = {"镜头": no, "url": mid["url"], "关键状态": mid.get("关键状态") or "",
                "应有": list(m.get("应有") or []), "应无": list(m.get("应无") or []),
                "序号": m.get("序号"), "叙事类型": m.get("叙事类型") or ""}
        if m:
            steps.append(item)
        elif mid["url"] not in states:
            states[mid["url"]] = {k: item[k] for k in ("关键状态", "应有", "应无")}
    if not steps:
        return states
    steps.sort(key=lambda x: (x["序号"] or 0, x["镜头"]))
    always = set(steps[0]["应无"])
    for s in steps[1:]:
        always &= set(s["应无"])
    keep = [steps[0]] if len(steps) == 1 else [steps[0], steps[-1]]
    for i, s in enumerate(keep):
        note = {"关键状态": s["关键状态"], "应有": s["应有"], "应无": s["应无"],
                "叙事类型": s["叙事类型"],
                "块内始终应无": [x for x in steps[0]["应无"] if x in always],
                "共几步": len(steps)}
        if len(keep) > 1:
            note["位置"] = "起始" if i == 0 else "结束"
        states.setdefault(s["url"], note)
    return states


def _piece_product_urls(product: dict, shot_refs: dict, shot_nos: list) -> tuple:
    """这一块要下发的商品图：按分镜级计划取并集（顺序 = 镜头顺序），返回（urls, 计划明细, 状态说明）。

    中间状态图排在商品原图前面：它是这一镜要锁死的关键状态，排前面 @图片1 就是它。
    装配序列整条落在一块里时按 _collapse_states 收敛成起始/结束两张，剩下的名额留给商品原图。
    计划里明确说这几镜不需要商品图（纯人物镜）时就真的不带；计划里根本没有这几镜
    （模型漏了、或没做计划）才退回全片共用的前几张。
    状态说明的值是 {"关键状态","应有","应无"}，装配序列另带 位置/块内始终应无/共几步：
    下游既要拿它写提示词，也要拿它做成片验收。
    """
    all_urls = product.get("image_urls") or []
    plan = (shot_refs or {}).get("分镜") or {}
    hit = [(no, plan[str(no)]) for no in shot_nos if str(no) in plan]
    if not hit:
        return all_urls[:rules.PIECE_REF_MAX], [], {}
    states = _collapse_states(hit)
    urls = []
    for _, p in hit:
        for u in p.get("urls") or []:
            if u not in urls and u not in states:
                urls.append(u)
    final = (list(states) + urls)[:rules.PIECE_REF_MAX]
    return final, [{"镜头": no, "参考图编号": p.get("参考图编号"), "理由": p.get("理由"),
                    "关键状态": p.get("关键状态") or "", "中间图": p.get("中间图") or {}}
                   for no, p in hit], {u: s for u, s in states.items() if u in final}
