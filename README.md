# ViralForge — 通用爆款短视频复刻 Agent

一句话:**给一条已验证的爆款参考视频 + 自己的商品信息(+ 可选自有素材),自动产出一条保留参考片结构、节奏与镜头语言的新成片。**

复刻不是"照着写一段提示词",而是:拆解参考片的镜头骨架 → 换成自己的人物/商品/场景重写剧本 → 优先复用用户自有素材、缺口才用 AIGC 补 → 拼接成片并处理音轨字幕。全程中间产物可追溯。

---

## 一、整体链路(一图看懂)

```text
输入:爆款参考视频(必选) + 商品图/名称/卖点 + 用户素材视频(可选) + BGM(可选)
  │
  ▼
┌─────────────────────────── pipeline.py 九步流水线 ───────────────────────────┐
│                                                                              │
│ ① ingest     素材入库校验     文件落任务目录,探测时长/分辨率,外挂工具探活      │
│ ② reference  爆款参考片拆解   VLM 拆分镜:景别/运镜/台词/转场/情绪/结构骨架    │
│ ③ audio      参考片音频判定   实测有没有 BGM/口播,决定成片音轨怎么处理        │
│ ④ materials  用户素材切片     把自有素材切成带时间码、语义标签的可召回片段     │
│ ⑤ product    商品事实卡       VLM 读商品图 + 用户填写 → 名称/卖点/人群        │
│ ⑥ script     剧本与分段       按 strict/creative 仿写剧本 + 人物/场景设定图,  │
│              ​                动态规划打包成 ≤15s 生成段(切点落在硬切镜头)   │
│ ⑦ match      分镜素材匹配     LLM 给每个分镜↔素材片段打分,按阈值分三路:      │
│              ​                直接裁剪 / 以素材为参考编辑生成 / 纯 AIGC 生成  │
│ ⑧ generate   分段出片(并发)  能裁的镜头裁用户素材,裁不动的交 seedance 生成;  │
│              ​                需要口播的片段用声音克隆(tts_clone)重新配音    │
│ ⑨ compose    拼接与音轨字幕   ffmpeg 归一化拼接 + 铺 BGM/原声 + 字幕烧录      │
│              ​                (可走 caption_clone 复刻参考片字幕样式)        │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
输出:output/tasks/{task_id}/ 下的 final.mp4 + 全部中间产物(JSON/MD/图/分段视频)
  │
  ▼
质检:review_case.py 成片自检(素材利用率 / 逐片状态 / 字幕一致性 / 门禁对账)
```

步骤顺序有一处刻意安排:**materials 在 product 之前**——用户没传商品图时,商品事实卡要从素材切片结论里抽商品帧。

每一步单独记状态、产物落盘,**幂等 + 断点续跑**:重跑任务时已完成的步骤直接复用产物;也可从任意步骤 reset 重跑(改一处不用全链路重来)。

---

## 二、怎么跑

| 入口 | 用途 | 用法 |
|---|---|---|
| **Web(正式入口)** | 建任务、传素材、跑流水线、看进度和产物 | `./start_web.sh`(默认端口 8420,建议配 `VF_WEB_TOKEN`) |
| CLI 调试 | 不起服务直接跑一条用例 | `python3 pipeline.py 17 --name "小米17 Pro Max"`(按 `materials/<编号>_*` 与 `videos/others/<编号>_*.mp4` 找素材;`--task <id> --from <step>` 可在已有任务上从某步重跑) |
| 单模块调试 | 只跑某一步 | `analyze_reference.py` / `analyze_materials.py` / `write_script.py` / `produce_video.py` 都能独立执行,参数在各自 `main()` 里改 |

Web 服务是 `server.py`(Flask API)+ `web/index.html`(单页前端),API 覆盖:建任务、上传素材(`reference_video` / `user_videos` / `product_images` / `person_images` / `bgm`)、后台跑流水线、从某步骤重跑、取产物文件。前端轮询 `task.json` 展示步骤进度与日志。

依赖极轻(`requirements.txt`):`requests`、`bce-python-sdk`、`imageio-ffmpeg`(自带 ffmpeg 二进制,无需系统安装)、`flask`。主链路**不需要 GPU**:模型调用全部走 API,本地只做 ffmpeg 剪辑拼接。

---

## 三、代码分层

### 1. 编排层(任务状态机)

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 只做编排:九步顺序、跑步骤记状态、失败停在当前步骤、断点续跑(`reset_from`)、CLI |
| `task_store.py` | 任务目录与素材登记:一个任务 = `output/tasks/{task_id}/`,参数、状态、日志、产物全落任务目录 |
| `media.py` | ffmpeg / ffprobe 薄封装:探时长、裁片段(含放慢补时长)、抽音轨、抽帧 |
| `server.py` + `web/` | Web API 与前端,后台线程调 `pipeline.run()` |
| `produce_video.py` | 双重角色:① 早期的批量端到端链路(按 `materials/{编号}/` 出片,不走素材匹配);② 提示词按 Skills 重写 + seedance 段生成 + ffmpeg 归一化工具 |

### 2. 每一步的实现

| 文件 | 职责 |
|---|---|
| `analyze_reference.py` | 参考片拆解:VLM 把视频拆成分镜 + 总结创意/结构/节奏/音乐情绪。字段名统一中文键 |
| `ref_audio.py` | 参考片音轨判定:有没有口播/BGM,按规则表决定整轨复用还是 demucs 分离出伴奏与纯人声 |
| `analyze_materials.py` | 用户素材切片:切成带时间范围、主体、动作、情绪、标签的语义片段,并生成「召回文本」备用于向量检索 |
| `product_images.py` | 商品参考图链路:每张图结构化理解 → 选基准图 → 判断并改造(一张一张生成并比对)→ 按分镜分配 → 中间状态图 |
| `product_facts.py` | 商品事实卡:用户填的信息 + 商品图 → name/卖点/外观/images,下游唯一商品口径 |
| `write_script.py` | 仿写剧本三步:换题材重写分镜(逐条带「复刻说明」可解释)→ 人物/场景设定图 → 动态规划分段(seedance 单次 ≤15s,切点避开叠化等软转场) |
| `shot_match.py` | 分镜 ↔ 素材匹配:候选打分、一个片段只服务一个分镜、三路分流 |
| `segment_build.py` | 分段出片:段内按镜头混合(能裁的裁素材,裁不动的交 seedance 补),补片走不通退回素材 |
| `voice_dub.py` | 音色基准与配音:优先带走用户原声,其次分离纯人声/AI 段提纯,VoxCPM2 克隆配音 |
| `compose_video.py` | 拼接与音轨字幕:归一化拼接、铺 BGM/原声、SRT/ASS 烧字幕、成片查画面残字 |
| `report.py` | 溯源报告:只读中间产物出 `report.md`(每张图/每段视频的原料、prompt、产物) |
| `aigc.py` | 模型调用统一入口:wenchain 网关(qwen LLM/VLM、seedream 生图、seedance 生视频)+ 独立 Gemini 网关(视频/音频理解,媒体 base64 内联);gemini 不通自动回落 qwen |
| `line_art.py` | 真人风控规避:seedance 拒收含真人人脸的参考图,改用「写实身体 + 面部黑白线稿」的生成图当参考,保住发型/服装/体型一致性 |
| `storage.py` | BOS 上传,把本地文件变成模型可访问的公网 URL |

### 3. 决策与质检层(判定口径的单一来源)

| 文件 | 职责 |
|---|---|
| `rules.py` | **规则是数据**:素材怎么用、音轨怎么处理等判定全部写成规则表,自上而下首条命中,末行兜底;`python3 rules.py` 渲染出 `output/rules.md` 总表 |
| `gates.py` | **标注是假设,用前实测**:VLM 说「有人声/有 BGM」只算候选,资产进成片前用 ffmpeg 实测(响度/有声占比/谱平坦度)过门禁;报告写进产物留痕;`python3 gates.py` 出 `output/gates.md` |
| `review_case.py` | 成片自检:`python3 review_case.py <task_id>`,只读产物 + 量音量,不调模型。查素材利用率、逐片状态、字幕重叠、音轨连续性,并对账「门禁没过却被使用」的资产 |

### 4. 外挂工具层(`Agent_tools/`)

重依赖(模型推理)隔离在子进程/独立环境,统一从 `registry.py` 接入:探活失败时对应步骤**明确跳过并记日志**,不拖垮整条链路。

| 工具 | 用途 | 用在哪一步 |
|---|---|---|
| `tts_clone` | 声音克隆(VoxCPM2):用参考音色为新台词配音 | generate(素材静音后重新配音) |
| `caption_clone` | 字幕风格克隆(词级 ASR + 样式提取):把参考片字幕的样式/位置/节奏搬到成片 | compose(copy_subtitles 开启时) |

模型权重统一从 `MODEL_ROOT`(config.env,默认 `/root/jmzhang/models`)下找。

### 5. 提示词规范(`Skills/`)

`SeedancePromptSkill.md` / `ImagePromptSkill.md` / `VideoPromptSkill.md`:生成前 LLM 会按这些规范把剧本的段提示词重写一遍(素材绑定 `@图片N`、镜头分拍、一镜一运镜、兜底约束包)。

---

## 四、关键机制说明

### 复刻参数(options)

| 参数 | 默认 | 含义 |
|---|---|---|
| `rewrite_mode` | `strict` | `strict` 一镜对一镜,只替换人物/商品/场景;`creative` 保留传播结构(钩子/节奏/反转位),允许改写剧情 |
| `use_user_materials` | `true` | 有用户素材时优先复用,不足才生成 |
| `copy_bgm` | `true` | 把参考片音乐铺到成片下(上传独立 BGM 优先用用户的;实测无音乐则不铺) |
| `copy_reference_audio` | `false` | 成片直接用参考片原声 |
| `copy_subtitles` | `false` | 按剧本台词生成字幕并烧进画面,可克隆参考片字幕样式 |

### 素材匹配三路分流(match → generate)

LLM 给「分镜 ↔ 素材片段」打匹配分(阈值在 `rules.py`):

- **≥ DIRECT_USE_SCORE**:直接裁剪用户素材(必要时最多放慢 1.6 倍补时长);
- **≥ EDIT_SCORE**:素材当参考片段做编辑生成(参考段需 >5s 才有约束力,凑不够改走参考图);
- **其余**:纯 AIGC 生成(seedance,参考图 = 线稿人物设定图 + 商品原图)。

### 真人人脸风控链路

seedance 拒收真人人脸参考图(错误码 40000002)。解法:VLM 先把原图人物外观读成文字 → 文生图画「写实身体 + 面部线稿」的替身图当参考 → 生成提示词前置真人化指令,成片里回到真人脸。`experiments/` 下的 `face_*_seedance.py`、`run_face_lineart_trials.py`、`skin_prompt_trials.py`、`eval_line_art.py` 都是这条链路的试验脚本,不在主链路上。

---

## 五、目录结构

```text
ViralForge/
├── pipeline.py              # 编排:九步顺序、步骤状态、幂等断点续跑、CLI
├── task_store.py            # 任务目录与素材登记
├── media.py                 # ffmpeg/ffprobe 薄封装(裁片段、抽帧、抽音轨)
├── server.py / web/         # Web API + 单页前端
├── start_web.sh             # Web 服务启停脚本
├── analyze_reference.py     # ② 参考片拆解
├── ref_audio.py             # ③ 参考片音轨判定与分离
├── analyze_materials.py     # ④ 用户素材切片
├── product_images.py        # ⑤ 商品参考图链路(理解/选图/改造/按镜分配/中间状态图)
├── product_facts.py         # ⑤ 商品事实卡
├── write_script.py          # ⑥ 仿写剧本 + 设定图 + 分段
├── shot_match.py            # ⑦ 分镜 ↔ 素材匹配
├── segment_build.py         # ⑧ 分段出片(裁素材 / 素材编辑 / 参考图生视频)
├── voice_dub.py             # ⑧ 音色基准与克隆配音
├── compose_video.py         # ⑨ 拼接 + 音轨 + 字幕
├── produce_video.py         # 批量端到端链路 + 提示词重写 + seedance 段生成
├── aigc.py                  # 模型调用(LLM/VLM/生图/生视频)
├── line_art.py              # 真人人脸风控规避
├── storage.py               # BOS 上传
├── rules.py                 # 判定规则表(规则是数据)
├── gates.py                 # 实测门禁(音轨等资产用前实测)
├── review_case.py           # 成片自检
├── report.py                # 溯源报告(每张图/每段视频的原料与 prompt)
├── config.py / config.env   # 网关地址、模型名、密钥等配置
├── Agent_tools/             # 外挂工具:tts_clone 声音克隆、caption_clone 字幕克隆
├── Skills/                  # 生成提示词规范(Seedance/Image/Video/商品参考图)
├── experiments/             # 一次性试验脚本(真人线稿、皮肤提示词等,不在主链路)
├── tests/                   # 冒烟自测脚本
├── materials/{编号}/        # 测试用商品素材(文件名即商品名)
├── videos/                  # 参考视频库
├── output/
│   ├── tasks/{task_id}/     # Web/pipeline 任务:task.json + 各步产物 + final.mp4 + report.md
│   ├── script_analysis/     # 参考片拆解产物(单模块跑法)
│   ├── script_writing/      # 剧本产物(单模块跑法)
│   ├── produced/            # produce_video 批量链路产物
│   ├── rules.md / gates.md  # 规则表与门禁清单(自动渲染)
│   └── ...
├── 通用爆款复刻Agent设计文档.md   # 设计蓝图(pipeline 是其 Phase 1+2 的落地)
├── 测试记录.md                    # 每轮测试的问题与修复记录
└── 视频理解维度分层报告.md        # 视频理解维度的调研报告
```

---

## 六、设计原则

1. **用户素材为主、AI 补片为辅**:段内可混合,只有素材覆盖不了的镜头才生成;素材利用率是成片主指标。
2. **判定口径单一来源**:决策进 `rules.py` 规则表、事实检查进 `gates.py` 门禁表,管线里不散写 if + 魔法数;改口径只动一处。
3. **显式失败,不静默丢内容**:工具不可用就明确跳过并记日志;门禁判不出走声明好的「放行/拦下」策略;失败原因写进 `task.json`。
4. **一切可追溯**:每步产物(拆解 JSON、剧本、匹配结果、门禁报告)落盘留痕,`review_case.py` 用同一把尺子复核每轮成片。
