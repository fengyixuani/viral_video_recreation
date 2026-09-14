# tts_clone 自测报告

跑法：`python tests/test_tts_clone.py`（本文件由脚本自动生成）

- 后端：`run_voxcpm2_zero_shot.py`，参考音一律抽成 16000Hz 单声道
- 参考音 A（正常，整片 mean -23.8dB）：`be0a0f489f48ad82_干发慕斯-素材1.MP4` 取 0.0-8.0s，原话「头发一油就紧贴头皮看着太显脸大了」
- 参考音 B（很轻，整片 mean -39.2dB）：`d3c75cf0ec9a3696_干发慕斯-素材4.MOV` 取 5.39-7.96s，原话「OK然后捏一捏那个泡沫」
- 「ASR 回听」= 用本地 Qwen3-ASR 把产出转写回来，校验**念的就是给的文案**（同音错字属 ASR 误识，不是合成错）

## 合成结果

**C1 默认(basic) + 短文案 14 字**

- 产物：`samples/C1_短文案.wav` | 时长 2.88s | 44100 Hz / 1 声道 | mean -19.1 dB / max -1.5 dB
- 给的文案：「洗完头喷一点，发根立马蓬起来」
- ASR 回听：「洗完头，喷一点，发根立马蓬起来。」
- 响度归一：是 | 耗时 35.7s（含模型加载）

**C2 默认(basic) + 长文案 29 字**

- 产物：`samples/C2_长文案.wav` | 时长 5.92s | 44100 Hz / 1 声道 | mean -17.8 dB / max -1.5 dB
- 给的文案：「头发一油就塌，抓两下发根，蓬松度立马回来，出门前三十秒搞定」
- ASR 回听：「头发一油就塌，抓两下发根，蓬松度立马回来。出门前三十秒搞定。」
- 响度归一：是 | 耗时 38.1s（含模型加载）

**C3 极轻参考音(-39.2dB) + 默认归一 -16 LUFS**

- 产物：`samples/C3_轻参考_已归一.wav` | 时长 3.36s | 44100 Hz / 1 声道 | mean -15.2 dB / max -2.1 dB
- 给的文案：「按一下不塌，支撑力真的够」
- ASR 回听：「看一下，不塌支撑力真的。」
- 响度归一：是 | 耗时 35.3s（含模型加载）

**C4 同 C3 但 lufs=off（对照：不归一有多轻）**

- 产物：`samples/C4_轻参考_未归一.wav` | 时长 2.4s | 48000 Hz / 1 声道 | mean -34.6 dB / max -20.3 dB
- 给的文案：「按一下不塌，支撑力真的够」
- ASR 回听：「按一下不塌，支撑力真的够。」
- 响度归一：否（lufs=off） | 耗时 34.0s（含模型加载）

**C5 prompt_mode=ultimate（对照：为什么默认不用它）**

- 产物：`samples/C5_ultimate模式.wav` | 时长 5.28s | 44100 Hz / 1 声道 | mean -17.2 dB / max -1.4 dB
- 给的文案：「洗完头喷一点，发根立马蓬起来」
- ASR 回听：「紧贴头皮，看着太显脸大了。洗完头喷一点，发根立马蓬起来。」
- 响度归一：是 | 耗时 44.6s（含模型加载）

## 错误路径（都应 ok=false + 中文原因，不抛异常）

- **E1 参考文件不存在** → `参考素材不存在：/nope.mp4`
- **E2 ultimate 模式缺 ref_text** → `ultimate 模式需要参考音转写 ref_text（或改用 basic 模式）`
- **E3 后端脚本路径错（未就绪）** → `后端未就绪：TTS_CLONE_PYTHON=/root/miniconda3/envs/voxcpm/bin/python TTS_CLONE_SCRIPT=/nope.py`

## 怎么听

`tests/samples/` 里放了参考原声和克隆产物，按对听：

- 音色像不像：`ref_loud_原声.wav` ↔ `C1_短文案.wav` / `C2_长文案.wav`
- 轻参考音也能听清：`ref_quiet_原声.wav` ↔ `C3_轻参考_已归一.wav`
- 为什么必须归一：`C3_轻参考_已归一.wav` ↔ `C4_轻参考_未归一.wav`（同文案同参考，只差 `lufs=off`）
- 为什么默认 basic：`C1_短文案.wav` ↔ `C5_ultimate模式.wav`（ultimate 会把参考音原话也念出来）

原始结果 JSON：

```json
[
  {
    "name": "C1",
    "cost_s": 35.7,
    "asr": "洗完头，喷一点，发根立马蓬起来。",
    "ok": true,
    "output": "/home/work/data/wanghequan/Agent_tools/tts_clone/tests/samples/C1_短文案.wav",
    "duration": 2.88,
    "lufs_normalized": true
  },
  {
    "name": "C2",
    "cost_s": 38.1,
    "asr": "头发一油就塌，抓两下发根，蓬松度立马回来。出门前三十秒搞定。",
    "ok": true,
    "output": "/home/work/data/wanghequan/Agent_tools/tts_clone/tests/samples/C2_长文案.wav",
    "duration": 5.92,
    "lufs_normalized": true
  },
  {
    "name": "C3",
    "cost_s": 35.3,
    "asr": "看一下，不塌支撑力真的。",
    "ok": true,
    "output": "/home/work/data/wanghequan/Agent_tools/tts_clone/tests/samples/C3_轻参考_已归一.wav",
    "duration": 3.36,
    "lufs_normalized": true
  },
  {
    "name": "C4",
    "cost_s": 34.0,
    "asr": "按一下不塌，支撑力真的够。",
    "ok": true,
    "output": "/home/work/data/wanghequan/Agent_tools/tts_clone/tests/samples/C4_轻参考_未归一.wav",
    "duration": 2.4,
    "lufs_normalized": false
  },
  {
    "name": "C5",
    "cost_s": 44.6,
    "asr": "紧贴头皮，看着太显脸大了。洗完头喷一点，发根立马蓬起来。",
    "ok": true,
    "output": "/home/work/data/wanghequan/Agent_tools/tts_clone/tests/samples/C5_ultimate模式.wav",
    "duration": 5.28,
    "lufs_normalized": true
  },
  {
    "name": "E1",
    "cost_s": 0.0,
    "asr": "",
    "ok": false,
    "error": "参考素材不存在：/nope.mp4"
  },
  {
    "name": "E2",
    "cost_s": 0.0,
    "asr": "",
    "ok": false,
    "error": "ultimate 模式需要参考音转写 ref_text（或改用 basic 模式）"
  },
  {
    "name": "E3",
    "cost_s": 0.0,
    "asr": "",
    "ok": false,
    "error": "后端未就绪：TTS_CLONE_PYTHON=/root/miniconda3/envs/voxcpm/bin/python TTS_CLONE_SCRIPT=/nope.py"
  }
]
```
