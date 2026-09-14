# tts_clone —— zero-shot 声音克隆工具

给一段**参考人声** + 一段**文案**，产出用这个音色念出文案的 wav。参考音不需要事先训练，
5~10 秒的干净人声就够（zero-shot / voice cloning）。

从 `Viral_Video_Agent` 的 `src/editing/tts.py` 抽出来的独立版本：调用方这一侧只依赖
**python3 标准库 + ffmpeg**，真正跑模型的后端在子进程里（默认 VoxCPM2，自带在
`backends/`，用它自己的 conda env）。

```
tts_clone/
├── tts_clone.py                        # 工具本体（库 + 命令行）
├── backends/run_voxcpm2_zero_shot.py   # 默认后端：VoxCPM2 零样本克隆
├── tests/test_tts_clone.py             # 自测：4 组真实合成 + 3 条错误路径
├── tests/TEST_REPORT.md                # 自测报告（脚本生成，含实测数据）
├── tests/samples/                      # 参考原声 + 克隆产物，可直接对听
└── README.md
```

## 输入 / 输出

**输入（4 个必填 + 2 个可选）**

- `ref` / `ref_audio`（必填）：参考人声文件。视频音频都行（`.mp4/.mov/.wav/.m4a`…），
  只要有音轨；内部会用 ffmpeg 抽成 16k 单声道 wav。
- `text`（必填）：要合成的目标文案。中文为主，标点会影响停顿。
- `out` / `out_wav`（必填）：输出 wav 路径，父目录自动创建。
- `ref_range`（可选）：`"起-止"` 秒，只取参考音的这一段，如 `"12.0-18.0"`。
  不给就用整段素材——素材很长时**建议给**，参考音里混进别人说话/环境噪声会污染音色。
- `ref_text`（可选）：参考音里实际说的原话。**只有 `prompt_mode=ultimate` 会用到**，
  默认的 basic 模式忽略它。
- `prompt_mode`（可选）：`basic`（默认）只把参考音波形喂给模型，产出就是目标文案；
  `ultimate` 把参考音 + 转写一起喂（理论上音色更贴），但实测输出不可控，见「踩过的坑」。
- `lufs`（可选）：本次的响度目标，默认 `-16`；传 `off` 保留模型原始电平。

**输出**

- 磁盘产物：`out` 指定的 wav —— **44100 Hz、单声道、16bit PCM**，整体响度归一到
  -16 LUFS（真峰值 -1.5 dB）。可以直接送进 ffmpeg 混音。
- 返回值 / stdout 的一行 JSON：
  - 成功：`{"ok": true, "output": "/tmp/out.wav", "duration": 6.08, "lufs_normalized": true}`
  - 失败：`{"ok": false, "error": "参考素材不存在：/nope.mp4"}`
  - `duration` 是产出音频的秒数（合成前**无法**精确预知，需要卡时长就按它裁/变速）。
  - 命令行下成功 exit code = 0，失败 = 1；失败不抛异常，`error` 是可直接展示的中文原因。

## 用法

命令行：

```bash
python tts_clone.py \
  --ref /path/素材1.MP4 --ref-range 0.0-8.0 \
  --text "洗完头喷一点，发根立马蓬起来" \
  --out /tmp/out.wav
# {"ok": true, "output": "/tmp/out.wav", "duration": 2.56, "lufs_normalized": true}
```

文案里有引号/换行时用 `--text-file 文案.txt` 代替 `--text`，`--quiet` 只打印结果 JSON。

Python 库：

```python
import sys
sys.path.insert(0, "/root/jmzhang/baidu/ViralForge/Agent_tools/tts_clone")
from tts_clone import clone, available

if not available():          # 后端解释器/脚本没配好时先给用户提示，别等到调用失败
    raise SystemExit("TTS 后端未就绪")

r = clone(ref_audio="/path/素材1.MP4", text="要念出来的文案",
          out_wav="/tmp/out.wav", ref_range="0.0-8.0")
if r["ok"]:
    print(r["output"], r["duration"])
else:
    print("失败：", r["error"])
```

## 前置条件

- `ffmpeg`（抽参考音 + 响度归一都靠它）。不在 PATH 上会自动回落到 `voxcpm` / `media` conda
  环境里那份；也可以 `export FFMPEG=/path/to/ffmpeg`。
- 后端 conda env：默认 `/root/miniconda3/envs/voxcpm/bin/python`，里面装了 `voxcpm`、
  `soundfile`。
- **权重（跨项目共用一份）**：后端按 `VOXCPM_MODEL_DIR` → `$MODEL_ROOT/VoxCPM2`
  （`MODEL_ROOT` 默认 `/root/jmzhang/models`）→ HF 仓库 id `openbmb/VoxCPM2` 的顺序找。
  本机 `/root/jmzhang/models/VoxCPM2` 已指向那份 4.9G 权重，**不会再走网络下载**
  （HF 缓存里那份是不全的，只有 364M，缺 `model.safetensors`，别指过去）。
- GPU 可选：有卡时单次合成占用不到 8G（源机器 L20）。**这台机器上 NVML 不可用（无卡/未透传），
  会退化到 CPU 推理**——能跑通，但慢：4.0s 音频实测 114s（含权重加载）。批量合成前先确认有卡。

## 环境变量

- `TTS_CLONE_PYTHON` / `TTS_CLONE_SCRIPT`：后端解释器 / 脚本。**换模型只改这两个**。
- `MODEL_ROOT`：共享模型根，默认 `/root/jmzhang/models`（多项目共用权重）。
- `VOXCPM_MODEL_DIR`：直接指定 VoxCPM2 权重目录，优先级高于 `MODEL_ROOT`。
- `TTS_CLONE_LUFS`：产出响度目标，默认 `-16`；`off` 关闭归一。
- `TTS_CLONE_TIMEOUT`：单次合成超时秒数，默认 600（首次要加载权重）。
- `TTS_CLONE_PROMPT_MODE`：`basic`（默认）/ `ultimate`，见「踩过的坑」。
- `TTS_CLONE_PROMPT_SR`：参考音采样率，默认 16000，**不建议改**（见下）。
- `FFMPEG`：ffmpeg 路径。
- 后端自己的可调项：`WHQ_VOXCPM_CFG`（默认 2.0，越大越贴参考音色、过大不稳）、
  `WHQ_VOXCPM_STEPS`（默认 30，越大越精细越慢）、`WHQ_VOXCPM_OUTPUT_SR`。

## 换后端

后端只需满足这个 CLI 契约，就能替换（CosyVoice3 的官方脚本天然满足）：

```
<python> <script> --prompt-wav P.wav --prompt-asr P.json --text "文案" --output OUT.wav
P.json = {"results": [{"text": "参考音转写"}]}
```

## 踩过的坑

- **别用 ultimate cloning（默认已关）**：把参考音转写一起喂给模型（`prompt_wav_path` +
  `prompt_text`），文档上说音色最贴，实测**输出不可控**：两次跑同一条 14 字文案，产出
  5.76s / 7.84s，ASR 回听是「参考音原话 + 目标文案」——参考那句被一起念了出来；换 29 字
  长文案则把开头「头发一油就塌，抓两下」吞掉改写成「锻炼一下」。同样两条用例在 basic 模式
  下都只念目标文案、字字对得上。所以默认 `prompt_mode=basic`，真要试 ultimate 一定要
  ASR 回听校验。
- **响度必须归一**：zero-shot 会把参考音的响度一起克隆。实测参考素材本身 mean -39 dB 时，
  产出的配音全在 -40 dB 左右，混进 BGM 后基本听不见人声。所以工具默认强制归一到 -16 LUFS，
  不要为了「保真」关掉它。
- **参考音喂 16k 就好，别改 44.1k**：VoxCPM2 的 AudioVAE 编码率就是 16000，48k 输出是
  生成式扩带宽、不吃参考音高频。实测喂 44.1k 产出反而更闷（>8kHz 能量占比 7.6% vs 16k 的
  15.5%），因为多了一道 44.1k→16k 重采样。
- **参考音要干净、单人、5~10 秒**：混进别人说话或现场噪声，音色会跑。长素材务必用
  `--ref-range` 截出一段。
- **弱参考音会让内容也不稳**：实测拿一段 2.57s、整片 -39dB、内容还是现场口令的素材当参考，
  同一条 12 字文案两次合成，一次逐字正确，一次 ASR 回听少了末字「够」。参考音够长够正常
  （正常口播 5~10s）时两条用例都逐字对得上。所以别拿边角料当参考音。
- **`lufs=off` 时输出是模型原生 48kHz**；走默认归一会统一成 44100Hz，方便直接混音。
- **时长不可控**：同一段文案每次合成的秒数会有出入（`duration` 是事后测的）。要卡画面时长，
  按返回的 `duration` 再决定裁剪 / `atempo` 变速，不要假设「几个字 = 几秒」。
- **每次调用都会重新加载模型**（一次 subprocess = 一次权重加载，约十几秒起）。批量合成时
  自己控制并发（同一块卡上别开太多），或改后端支持多条文案一次跑完。
- **PYTHONPATH 会污染后端**：工具内部已经把 `PYTHONPATH/PYTHONHOME/PYTHONSTARTUP`
  从子进程环境里清掉了，否则后端会串到调用方的 site-packages 报 `ModuleNotFoundError`。
