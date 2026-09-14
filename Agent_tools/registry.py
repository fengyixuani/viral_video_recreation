"""Agent_tools 接入层：外挂工具统一从这里进 pipeline，别处不要再 sys.path.insert。

接入规范（新工具照此形态挂进来）：
- 每个工具一个目录，`<name>.py` 同时是库和 CLI；重依赖（模型推理）隔离在子进程 /
  独立 conda env 里，调用方这一侧只依赖标准库 + ffmpeg；
- 返回统一为 dict：{"ok": True, ...} / {"ok": False, "error": "可直接展示的中文原因"}，
  不抛栈；CLI 成功 exit 0、失败 exit 1；
- 必须提供 available() 探活；未就绪时 pipeline 明确跳过并记日志，不崩整条链路；
- 配置只走环境变量，模型权重统一从 MODEL_ROOT（config.env）下找。

这里负责三件事：挂 sys.path、灌共享配置（MODEL_ROOT / VF_GPU）、探活。
"""
import os
import sys

import config  # noqa: F401  # 先把 config.env 灌进 os.environ（MODEL_ROOT / VF_GPU / ASR_* 都在里面）

_ROOT = os.path.dirname(os.path.abspath(__file__))

# 模型权重共享根：tts_clone（VoxCPM2）与 caption_clone 的词级 ASR（Qwen3-ASR /
# ForcedAligner）默认都从它下面找同名目录。
os.environ.setdefault("MODEL_ROOT", "/root/jmzhang/models")

# 推理钉在哪张卡（config.env 的 VF_GPU，留空 = 不限制）。两个工具的模型都在子进程里跑，
# 子进程继承这里设置的 CUDA_VISIBLE_DEVICES。
_gpu = os.getenv("VF_GPU", "").strip()
if _gpu and not os.getenv("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

for _tool in ("tts_clone", "caption_clone", "firered_asr"):
    _path = os.path.join(_ROOT, _tool)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import firered_asr  # noqa: E402  # pyright: ignore[reportMissingImports]
import tts_clone  # noqa: E402  # pyright: ignore[reportMissingImports]

# caption_clone 依赖 requests 等第三方包，缺了不该把 pipeline 一起拖死：
# 导入失败就置 None，caption_available() 会带上原因。
try:
    from caption_clone import clone_captions  # pyright: ignore[reportMissingImports]
    from core import asr_tokens as _asr_tokens  # pyright: ignore[reportMissingImports]
    CAPTION_ERROR = ""
except Exception as _exc:  # noqa: BLE001
    clone_captions = None
    _asr_tokens = None
    CAPTION_ERROR = "caption_clone 导入失败：%s" % _exc


def caption_available() -> bool:
    """字幕风格克隆是否可用（不含词级 ASR：ASR 缺失只是逐字时间退化，流程照跑）。"""
    return clone_captions is not None


def prewarm_asr() -> bool:
    """提前拉起 FireRedASR 常驻 worker（非阻塞）。

    模型加载十几秒，而补片校对是在 generate 步才用到的。任务一开始就拉起来，
    到用的时候模型已经热了；拉不起来也只是那一步退回不校对，不影响别的步骤。
    """
    return firered_asr.available() and firered_asr.start()


def probe() -> dict:
    """探活全部外挂工具，任务开始时调一次写进日志，别等到最后一步才发现后端没配。"""
    return {
        "tts_clone": tts_clone.available(),
        "caption_clone": caption_available(),
        "caption_asr": bool(_asr_tokens and _asr_tokens.available()),
        "firered_asr": firered_asr.available(),
    }
