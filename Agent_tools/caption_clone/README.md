# caption_clone —— 字幕特效克隆

给一条成片烧上**和参考视频同风格**的字幕：分析参考视频的字幕风格（文字颜色/字号/位置/
斜排/关键词高亮/特效偏好），再把这条成片的念白按同样的风格拆块、配色、配位、烧录。

从 `Viral_Video_Agent` 的 `src/editing/whq_clone/captions_clone` 抽出来的独立版本：
去掉了对 Agent 内部模块（`_common` / `pipeline_utils` / `as_core` / `obs`）的依赖，
LLM/VLM 和词级 ASR 都改成环境变量配置，可以单独跑、也可以被别的 Agent 当 tool 挂。

```
caption_clone/
├── caption_clone.py       # 对外入口（库 clone_captions() + 命令行）
├── core/                  # 实现模块（分工见 core/__init__.py）
│   ├── gateway.py         # LLM/VLM 网关 + JSON 兜底 + 阶段缓存（替代 pipeline_utils/as_core）
│   ├── _env.py            # FFMPEG/FFPROBE（替代 _common）
│   ├── asr_tokens.py      # 词级 ASR 子进程
│   ├── ref_analyzer.py / color_calib.py / style_profile.py / profile_cache.py   # 参考风格
│   ├── charstream.py / target_match.py                                          # 念白与编排
│   └── ass_burn.py / frames.py / inventory_md.py / style_spec.py                # 抽帧与烧录
└── README.md
```

## 输入 / 输出

**输入**

- `video`（必填）：目标成片，要烧字幕的那条视频。
- `ref_video`（可选但强烈建议）：参考视频，字幕风格的来源。不给、或文件不存在时用内置
  回退风格（白字口播 + 红橙斜排大字），流程照跑。
- `out_video`（可选）：输出路径，默认 `<成片名>_capfx.mp4`。
- `items`（可选）：字幕文本清单 `[{"start": 秒, "end": 秒, "text": "这一句念白"}]`。
  **有配音 plan 就传**：字幕文本用它，字字准确；不传就完全按成片 ASR 的识别结果做字幕，
  会带同音错字（工具内有一道等长纠错，但只能救回一部分）。
- `work_dir`（可选）：中间产物目录，默认 `<成片名>_capwork`。
- `typo_fix`（可选，默认开）：同音错字纠错，只接受**等长**改写，不会动时间轴。
- `llm_model` / `ref_fps`（可选）：覆盖拆块用的文本模型；参考视频抽帧率（默认 1.0 帧/秒，
  越大越准越慢越贵）。

**输出**

- 磁盘产物：
  - `out_video`：烧好字幕的成片（视频重编码，音轨 copy）。
  - `work_dir/目标字幕清单.md`：每个字幕块的时间/文字/颜色/位置/特效，人可读，出问题先看它。
  - `work_dir/ref_style/<参考名>_style_profile.json`：参考风格档（**按参考视频指纹缓存**，
    同一条参考视频第二次跑直接复用，省掉抽帧+VLM 的几分钟）。
  - `work_dir/ref_style/<参考名>_字幕清单.md`：参考视频里逐条字幕的识别结果。
  - `work_dir/final_asr/all_source_asr.json`：成片词级 ASR（逐字时间戳，同样带指纹缓存）。
  - `work_dir/captions_clone.ass`：实际烧录用的 ASS（可手改后自己 ffmpeg 重烧）。
- 返回值 / stdout 的一行 JSON：
  - 成功：`{"ok": true, "output": "...", "blocks": 24, "lines": 12, "work_dir": "...",
    "caption_md": "...", "profile": {"density": "high", "styles": ["narration", ...]},
    "cost_s": 210.5}`
  - 失败：`{"ok": false, "error": "拿不到任何念白句：...", "work_dir": "..."}`
  - 命令行成功 exit 0、失败 exit 1；任何异常都转成 `error` 字段，不抛栈。

## 用法

命令行：

```bash
python caption_clone.py \
  --video /path/成片.mp4 \
  --ref   /path/参考爆款.mp4 \
  --out   /path/成片_capfx.mp4        # 可省，默认 <成片名>_capfx.mp4
# {"ok": true, "output": "...", "blocks": 24, "lines": 12, "cost_s": 210.5, ...}
```

有配音 plan 时把文本喂进去（字幕更准）：

```bash
cat > items.json <<'JSON'
[{"start": 0.0, "end": 3.2, "text": "头发一油就紧贴头皮，看着太显脸大了"},
 {"start": 3.2, "end": 6.0, "text": "那就试试这瓶慕斯"}]
JSON
python caption_clone.py --video 成片.mp4 --ref 参考.mp4 --items items.json
```

Python 库：

```python
import sys
sys.path.insert(0, "/root/jmzhang/baidu/ViralForge/Agent_tools/caption_clone")
from caption_clone import clone_captions

r = clone_captions(video="/path/成片.mp4", ref_video="/path/参考爆款.mp4",
                   items=[{"start": 0.0, "end": 3.2, "text": "头发一油就紧贴头皮"}])
if r["ok"]:
    print(r["output"], r["blocks"], "块")
else:
    print("失败：", r["error"], "| 中间产物看：", r["work_dir"])
```

## 前置条件

- **带 libass 且带 libx264 的 ffmpeg**：libass 烧字幕、libx264 重编码输出，**缺一个都不行**。
  按「`FFMPEG` 环境变量 → PATH → `voxcpm`/`cutclaw`/`liveclip_vocal`/`storyline` conda 环境」
  的顺序找，并对候选实测这两项能力。注意 conda 的 `media` 环境是 `--disable-gpl` 构建，
  有 libass 但**没有 libx264**，选它会走到最后一步烧录才报 `Unknown encoder 'libx264'`，
  所以不在候选里。启动时会自检，不满足直接报错。
- **中文字体**：ASS 默认字体 `Noto Sans CJK SC`（`WHQ_CAPTION_FONT` 可换）。字体缺失时
  libass 会回落到默认字体，中文可能变方框。
- **LLM/VLM 网关**：`WENCHAIN_BASE_URL` + `WENCHAIN_API_KEY`（默认沿用源项目的内网网关与
  租户标识）。参考风格分析用视觉模型、拆块用文本模型，都是网络调用。
- **词级 ASR（可选但影响很大）**：`ASR_PYTHON` + `ASR_SCRIPT`，默认按候选顺序找本机那套
  qwen3-asr（`/root/chengzhiyang/miniconda3/envs/qwen3-asr-cu128` +
  `/root/chengzhiyang/Viral_Video_Split/understanding/batch_qwen3_asr.py`，权重同目录的
  `models/Qwen3-ASR-0.6B`）。缺了就只能按 `items` 的窗口线性铺字，逐字时间不准；
  既没 ASR 又没 `items` 时会直接返回失败（没有文本来源）。
- Python 包：`requests` 必需；`Pillow`+`numpy`（像素取色校准，缺了跳过校准只用 VLM 目测色）、
  `jieba`（词边界，缺了只用标点断句）、`json_repair`（LLM JSON 兜底）都是可选。

## 环境变量

样式与流程（沿用源项目的 `WHQ_CAPTION_*` 命名，方便和 Agent 里的行为对齐）：

- `WHQ_CAPTION_REF_FPS=1.0`：参考视频抽帧率。
- `WHQ_CAPTION_VLM_MODEL` / `WHQ_CAPTION_LLM_MODEL`：读参考帧的视觉模型 / 拆块的文本模型。
- `WHQ_CAPTION_PROFILE`：直接指定一个成品 `style_profile.json`，**跳过参考分析**（最省钱的复用方式）。
- `WHQ_CAPTION_PROFILE_DIR`：参考风格缓存目录，指到一个公共目录可以跨任务共享。
- `WHQ_CAPTION_FONT`：ASS 主字体，默认 `Noto Sans CJK SC`。
- `WHQ_CAPTION_REVEAL=block|char`：整块弹出（默认）/ 逐字揭示。
- `WHQ_CAPTION_ANIM=off|on`：入场动画（默认 off，直接展示）。
- `WHQ_CAPTION_HOT_SCALE=1.25`：块内关键词相对正文的放大倍数（1.0 = 只换色不变大）。
- `WHQ_CAPTION_PLAIN=0`：普通字幕模式。开（`1`）则全片只有一种样式——所有块降级成口播档，
  块内关键词不变色不放大、不斜排、不加特效、一律底部居中；参考片的字体/字号/正文色照常复刻。
  要「每条片子字幕风格都一致」时开它。
- `WHQ_CAPTION_POS=bottom|mixed`：全部底部居中（默认）/ 彩色大字上顶部。
- `WHQ_CAPTION_TYPO_FIX=1`：同音错字纠错（`--no-typo-fix` 等价于 0）。
- `CAPTION_RESUME=0`：忽略所有阶段缓存，强制重跑参考分析与 ASR。

外部依赖路径：`FFMPEG` / `FFPROBE`、`ASR_PYTHON` / `ASR_SCRIPT`、
`QWEN3_ASR_MODEL` / `QWEN3_FORCED_ALIGNER`（默认取 `$MODEL_ROOT`，即 `/root/jmzhang/models`
下的同名目录，没有再回落到 Split 仓的 `models/`）、
`WENCHAIN_BASE_URL` / `WENCHAIN_API_KEY`、`TEXT_LLM_MODEL` / `VISION_LLM_MODEL`、
`WHQ_LLM_CONCURRENCY=3`（VLM 分批并发数）。

## 与 Agent 里那份的差异

- 只保留**本机 ASS 烧录**后端。源项目还有一个「美摄云端花字」后端（`captions_clone/meishe`），
  它依赖内网 BOS/美摄的账号与 SDK，搬出来对外部使用者没意义，所以没有拷过来。
- LLM/VLM 不再走 Agent 的 `as_core`（langchain + 异步），改成 `core/gateway.py` 里的
  requests 直连；模型名、超时、重试、并发都由环境变量控制。
- 词级 ASR 不再从 `_common.REPO` 推路径，改成 `ASR_PYTHON` / `ASR_SCRIPT` 直接指。
- 入口从「失败就静默返回 None、由调用方回退旧字幕」改成**返回带 `error` 的 dict**，
  让调用方能知道为什么没成。

## 踩过的坑

- **ffmpeg 必须带 libass**：imageio-ffmpeg 那类静态包常常不带，烧录会失败或悄悄出一条没字幕的片子。
  工具启动时自检并直接报错。
- **参考风格分析很贵**（抽帧 + 多批 VLM + 像素取色），所以按「参考视频指纹 + 抽帧率 + 模型」
  缓存。同一条参考视频复用时把 `WHQ_CAPTION_PROFILE_DIR` 指到公共目录，或直接用
  `WHQ_CAPTION_PROFILE` 指成品 profile。
- **字幕文本尽量用 `items`**：只靠 ASR 会有同音错字（实测「壁垒→避雷」这类）。工具里的纠错
  只接受等长改写，救不了所有情况。
- **`items` 的文本要是「实际念出来的话」**：拿分镜脚本/DNA 文案去当 items，会出现字幕和人声
  对不上——ASR 流里定位不到这句，就退化成按窗口线性铺字，时间轴会漂。
- 中间产物别删：出问题先看 `work_dir/目标字幕清单.md`（拆块/配色/特效是否合理）和
  `work_dir/ref_style/<参考名>_字幕清单.md`（参考视频到底被读成了什么）。

## 实测

在 `agentcut_20260813_111625_loop1.mp4`（36.7s 成片）+ 参考 `f205d04011a08722_干发.mp4` 上
跑通：`ok=true`，8 句念白 → 9 个字幕块，全程 93.6s（含参考分析 + 成片词级 ASR + LLM 拆块 +
烧录）。输出时长 36.70s（原 36.69s）无截断，t=5.0s 处画面下三分之一有 1.2 万个像素变化
（正是那一块 `4.96-6.62 看着太显…`），字体解析到 `NotoSansCJK-Bold`。
