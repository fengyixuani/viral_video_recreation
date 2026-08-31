"""asr_tokens —— 词级（逐字）ASR 子步骤。字幕的逐字揭示时间全靠它。

原 whq_clone 版本从 `_common.REPO` 里找 Split 仓的 `understanding/batch_qwen3_asr.py`。
抽成独立工具后改成两个环境变量直接指：

    ASR_PYTHON  跑 ASR 的解释器（默认按候选顺序找本机的 qwen3-asr 环境）
    ASR_SCRIPT  批量词级 ASR 脚本（默认按候选顺序找 Split 仓的 batch_qwen3_asr.py）
    ASR_ENTRY   批量脚本内部调的单文件推理入口（默认 Split 仓 vendor 下的 run_qwen3_asr_test.py）

脚本契约（保持与原版一致，靠环境变量传参）：
    NARIS_SOURCE_GLOB=<视频或 glob>  NARIS_ASR_DIR=<缓存目录>
    -> 在 NARIS_ASR_DIR 下产出 all_source_asr.json（每条记录带 asr_items 逐字时间戳）

ASR 不可用时返回 None：上层会退化为「按 tts plan 窗口线性铺字」，字幕仍能烧上，
只是逐字时间不准；完全没有文本来源时（无 plan 又无 ASR）才会跳过字幕。
"""
import glob
import json
import os
import subprocess

from ._env import FFMPEG
from .gateway import file_fingerprint, fingerprint

def _first_path(env_name, *candidates):
    """环境变量优先，否则返回第一个存在的候选；都不存在时返回第一个候选（让 available() 报它）。"""
    val = os.getenv(env_name)
    if val:
        return val
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def _first_existing(env_name, *candidates):
    """同上，但都不存在时返回空串（权重路径这种「拿不准就别透传，让脚本用自己的默认」的场景）。"""
    val = os.getenv(env_name)
    if val:
        return val
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return ""


# 本机这套 ASR（解释器 + 脚本 + 权重）目前只有 chengzhiyang 目录下一份，跨项目共用它。
ASR_PYTHON = _first_path(
    "ASR_PYTHON",
    "/root/chengzhiyang/miniconda3/envs/qwen3-asr-cu128/bin/python",
    "/root/miniconda3/envs/qwen3-asr-cu128/bin/python",
)
ASR_SCRIPT = _first_path(
    "ASR_SCRIPT",
    "/root/chengzhiyang/Viral_Video_Split/understanding/batch_qwen3_asr.py",
)
# 权重目录：脚本自己也有默认值，这里只在显式配置（或共享模型根下存在）时透传
QWEN3_ASR_MODEL = _first_existing(
    "QWEN3_ASR_MODEL",
    os.path.join(os.getenv("MODEL_ROOT", "/root/jmzhang/models"), "Qwen3-ASR-0.6B"),
    "/root/chengzhiyang/Viral_Video_Split/models/Qwen3-ASR-0.6B",
)
QWEN3_FORCED_ALIGNER = _first_existing(
    "QWEN3_FORCED_ALIGNER",
    os.path.join(os.getenv("MODEL_ROOT", "/root/jmzhang/models"), "Qwen3-ForcedAligner-0.6B"),
    "/root/chengzhiyang/Viral_Video_Split/models/Qwen3-ForcedAligner-0.6B",
)
# ASR_SCRIPT 在两层里同名不同义：我们这层指批量脚本(batch_qwen3_asr.py)，批量脚本里指单文件
# 推理入口(run_qwen3_asr_test.py)。子进程直接继承我们的 ASR_SCRIPT 会让批量脚本把**自己**当
# 内层入口反复自我拉起 —— 实测堆出几百个 0% CPU 的空转进程、永远到不了模型（也就永远看不到
# 显卡动）。所以透给子进程前必须把它换成真正的内层入口。
ASR_ENTRY = _first_existing(
    "ASR_ENTRY",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ASR_SCRIPT))),
                 "common", "vendor", "Viral_Video", "run_qwen3_asr_test.py"),
)


def available():
    """词级 ASR 依赖（解释器 + 脚本）是否就位。"""
    return bool(os.path.exists(ASR_PYTHON) and os.path.exists(ASR_SCRIPT))


def _asr_env(cache_dir):
    """构造 ASR 子进程环境：缓存目录、内层单文件入口、numba 可写缓存、模型覆盖。"""
    env = os.environ.copy()
    env["NARIS_ASR_DIR"] = cache_dir
    env.setdefault("FFMPEG", FFMPEG)
    if ASR_ENTRY:                      # 换成内层单文件入口，避免批量脚本自我递归
        env["ASR_SCRIPT"] = ASR_ENTRY
    else:
        env.pop("ASR_SCRIPT", None)    # 找不到就让批量脚本用它自己的默认值
    # librosa 里有 numba njit(cache=True) 的函数，默认把缓存写到 site-packages 旁边；ASR env 装在
    # 别人家目录下写不进去，会直接抛 "no locator available" 让整条 ASR 失败。给它一个可写目录。
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(cache_dir, "numba_cache"))
    if QWEN3_ASR_MODEL:
        env["QWEN3_ASR_MODEL"] = QWEN3_ASR_MODEL
    if QWEN3_FORCED_ALIGNER:
        env["QWEN3_FORCED_ALIGNER"] = QWEN3_FORCED_ALIGNER
    return env


def _fp_path(out_json):
    """缓存指纹文件路径（``<json>.fp``）。"""
    return out_json + ".fp"


def _cache_valid(out_json, fp):
    """缓存是否命中：结果文件存在且记录的指纹与当前一致。"""
    if not os.path.exists(out_json):
        return False
    try:
        with open(_fp_path(out_json), encoding="utf-8") as f:
            return f.read().strip() == fp
    except OSError:
        return False


def _mark_cache(out_json, fp):
    """把当前指纹写进 .fp 标记缓存有效（写失败静默，代价只是下次重跑）。"""
    try:
        with open(_fp_path(out_json), "w", encoding="utf-8") as f:
            f.write(fp)
    except OSError:
        pass


def run_asr(source_glob, cache_dir):
    """对 ``source_glob`` 匹配的视频跑词级 ASR，产出 ``cache_dir/all_source_asr.json``。

    输入指纹（路径+大小+mtime）未变则复用缓存 —— ASR 很重（几十秒）；换了视频自动重跑。
    返回 json 路径；环境缺失/失败返回 None。
    """
    os.makedirs(cache_dir, exist_ok=True)
    out_json = os.path.join(cache_dir, "all_source_asr.json")
    paths = sorted(glob.glob(source_glob)) or (
        [source_glob] if os.path.exists(source_glob) else [])
    fp = fingerprint(file_fingerprint(paths))
    if _cache_valid(out_json, fp):
        return out_json
    if not available():
        print("[asr_tokens] 无 ASR 环境/脚本(ASR_PYTHON/ASR_SCRIPT), 跳过词级 ASR", flush=True)
        return None
    env = _asr_env(cache_dir)
    env["NARIS_SOURCE_GLOB"] = source_glob
    try:
        print("[asr_tokens] 跑词级 ASR: {} -> {}".format(source_glob, out_json), flush=True)
        subprocess.run([ASR_PYTHON, ASR_SCRIPT],
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(ASR_SCRIPT))),
                       env=env, check=True,
                       timeout=float(os.getenv("ASR_TIMEOUT", "900")))
    except Exception as exc:  # noqa: BLE001
        print("[asr_tokens] ASR 失败(降级): {}".format(str(exc)[:200]), flush=True)
        return None
    if not os.path.exists(out_json):
        return None
    _mark_cache(out_json, fp)
    return out_json


def items(video, cache_dir):
    """视频的逐字 asr_items 列表（合并所有记录）。失败返回 []。"""
    out_json = run_asr(video, cache_dir)
    if not out_json:
        return []
    try:
        with open(out_json, encoding="utf-8") as f:
            records = json.load(f) or []
    except (OSError, ValueError) as exc:
        print("[asr_tokens] ASR 结果解析失败: {}".format(str(exc)[:160]), flush=True)
        return []
    out = []
    for r in records:
        out.extend(r.get("asr_items") or [])
    return out
