"""firered_asr —— FireRedASR2 语音识别（带词级毫秒时间戳），给补片做「台词说到哪」的判据。

本模块只依赖标准库 + ffmpeg：模型/引擎跑在 worker.py 子进程里（见那里的协议与后端说明）。
worker 常驻，加载只发生一次，所以务必在流水线早期调一次 start() 预热，别等 generate
步才拉起来（加载二十几秒，逐块起进程会白等一分钟）。

默认走 TensorRT 后端（本机已建好引擎，热态单块 0.09s）；引擎失效或换了 GPU 架构时
把 FIRERED_BACKEND 改成 torch 即可退回 PyTorch，其余调用方式完全不变。

用法：
    import firered_asr
    firered_asr.start()                       # 非阻塞，尽早调
    r = firered_asr.transcribe("a.mp4")       # 任何带音轨的媒体都行，内部转 16k 单声道
    r["words"] -> [{"start": 0.49, "end": 0.61, "text": "睡"}, ...]

配置（都可用环境变量覆盖，默认值指向本机已装好的那份）：
    FIRERED_BACKEND    trt（默认）/ torch
    FIRERED_PYTHON     跑 worker 的解释器（trt 必须是 firered-trt env 那个）
    FIRERED_TRT_ENV    firered-trt env 根目录，用来拼 LD_LIBRARY_PATH
    FIRERED_SPEED_REPO firered_speed 仓库根（TensorRT 链路与引擎都在它下面）
    FIRERED_REPO       FireRedASR2S 仓库根（torch 后端用）
    FIRERED_MODELS     权重根（Punc 词表两个后端都要用）
    FIRERED_USE_GPU    0 关 GPU（只对 torch 后端有意义）
"""
import json
import os
import select
import subprocess
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))

READY_TIMEOUT = float(os.getenv("FIRERED_READY_TIMEOUT", "300"))   # 等引擎/权重加载
CALL_TIMEOUT = float(os.getenv("FIRERED_CALL_TIMEOUT", "120"))     # 等单条识别


def _first_exec(*candidates):
    """返回第一个可执行的候选；都不行就返回第一个，让报错带上原路径。"""
    for cand in candidates:
        if os.path.isabs(cand):
            if os.access(cand, os.X_OK):
                return cand
        else:
            from shutil import which
            if which(cand):
                return cand
    return candidates[0]


FFMPEG = os.getenv("FFMPEG") or _first_exec(
    "ffmpeg",
    "/root/miniconda3/envs/media/bin/ffmpeg",
    "/root/miniconda3/envs/voxcpm/bin/ffmpeg",
)
BACKEND = (os.getenv("FIRERED_BACKEND") or "trt").strip().lower()
TRT_ENV = os.getenv("FIRERED_TRT_ENV", "/root/miniconda3/envs/firered-trt")
SPEED_REPO = os.getenv("FIRERED_SPEED_REPO", "/home/work/chengzhiyang/firered_speed")
REPO = os.getenv("FIRERED_REPO", "/home/work/chengzhiyang/asr_test/FireRedASR2S")
MODELS = os.getenv("FIRERED_MODELS", "/home/work/chengzhiyang/asr_test/models")
# trt 后端必须用 firered-trt 那个解释器（TensorRT 10.10.0.31 + TensorRT-LLM 0.20.0）；
# torch 后端用 base conda（torch 2.6.0+cu124，依赖齐全）。
PYTHON = os.getenv("FIRERED_PYTHON") or _first_exec(
    os.path.join(TRT_ENV, "bin", "python") if BACKEND == "trt"
    else "/root/miniconda3/bin/python")

_ENGINE_DIR = os.path.join(SPEED_REPO, "tensorrt", "FireRedASR2-AED-TensorRT")
_AUX_DIR = os.path.join(_ENGINE_DIR, "aux_engine_float16")

_LOCK = threading.Lock()       # 补片是并发跑的（seg_workers 默认 5），worker 只有一条管道
_PROC = None
_READY = None                  # None=还没确认；dict=ready 结果（含失败原因）


def _need() -> list:
    """这个后端跑起来必须存在的东西，按「先报最可能缺的」排序。"""
    if BACKEND == "trt":
        return [PYTHON,
                os.path.join(SPEED_REPO, "tensorrt_full_pipeline.py"),
                os.path.join(_ENGINE_DIR, "encoder.plan"),
                os.path.join(_AUX_DIR, "fireredvad_vad.plan"),
                os.path.join(_AUX_DIR, "fireredvad_aed.plan"),
                os.path.join(SPEED_REPO, "tensorrt", "FireRedPunc", "punc_core.plan"),
                os.path.join(MODELS, "FireRedPunc")]
    return [PYTHON, os.path.join(REPO, "fireredasr2s"),
            os.path.join(MODELS, "FireRedASR2-AED", "model.pth.tar"),
            os.path.join(MODELS, "FireRedPunc", "model.pth.tar"),
            os.path.join(MODELS, "FireRedVAD", "VAD", "model.pth.tar")]


def available() -> bool:
    """后端要用的解释器 / 引擎 / 权重是否都在位。不在位时 transcribe 回明确 error，不抛。"""
    return all(os.path.exists(p) for p in _need())


def missing() -> str:
    """available() 为假时，说清缺哪一件（写进日志用）。"""
    for p in _need():
        if not os.path.exists(p):
            return "%s 后端缺 %s" % (BACKEND, p)
    return ""


def start() -> bool:
    """拉起常驻 worker，立刻返回（不等加载完）。已在跑就什么都不做。

    返回「有没有一个活着的 worker」，不代表模型已经就绪——就绪与否由 ready() 确认。
    """
    global _PROC
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            return True
        if not available():
            return False
        env = dict(os.environ, FIRERED_BACKEND=BACKEND, FIRERED_REPO=REPO,
                   FIRERED_MODELS=MODELS, FIRERED_SPEED_REPO=SPEED_REPO,
                   PYTHONUNBUFFERED="1")
        if BACKEND == "trt":
            # 缺这一条会报 libpython3.12.so.1.0 / libmpi.so 找不到，且是加载期才炸
            lib = os.path.join(TRT_ENV, "lib")
            env["LD_LIBRARY_PATH"] = ":".join(
                [lib] + [p for p in [os.environ.get("LD_LIBRARY_PATH")] if p])
        _PROC = subprocess.Popen(
            [PYTHON, os.path.join(_HERE, "worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        return True


def _readline(timeout: float) -> str:
    """带超时地读一行。worker 卡死时不能把整条流水线一起拖住。"""
    if _PROC is None or _PROC.stdout is None:
        return ""
    if not select.select([_PROC.stdout], [], [], timeout)[0]:
        return ""
    return _PROC.stdout.readline()


def ready() -> dict:
    """确认（必要时等待）worker 的加载结果，返回 {"ok":bool,"load_sec","backend"/"error"}。"""
    global _READY
    if _READY is not None:
        return _READY
    if not start():
        _READY = {"ok": False, "error": "后端未就绪：%s" % (missing() or "未知")}
        return _READY
    line = _readline(READY_TIMEOUT)
    if not line:
        _READY = {"ok": False, "error": "worker %.0fs 内没有报就绪" % READY_TIMEOUT}
        return _READY
    try:
        got = json.loads(line)
    except ValueError:
        _READY = {"ok": False, "error": "worker 首行不是 json：%s" % line[:120]}
        return _READY
    _READY = ({"ok": True, "load_sec": got.get("load_sec"),
               "backend": got.get("backend") or BACKEND} if got.get("ready")
              else {"ok": False, "error": got.get("error") or "worker 报未就绪"})
    return _READY


def _to_wav(src: str, dst: str) -> bool:
    """转 16k 单声道 PCM16。FireRed 对采样率和声道数都是硬断言，不转必崩。"""
    ret = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src, "-vn",
                          "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst],
                         capture_output=True, text=True)
    return ret.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 1024


def transcribe(media: str, req_id: str = "req") -> dict:
    """识别 media 的人声，返回 {"ok",...,"text","sentences","words","dur_s"}。

    media 可以是 mp4/wav/m4a…任何 ffmpeg 能读且带音轨的文件。没有音轨时 ffmpeg 转不出
    wav，回 ok=False（调用方据此当「这一片没有人声」处理，不要当成识别故障）。
    """
    if not media or not os.path.isfile(media):
        return {"ok": False, "error": "文件不存在：%s" % media}
    st = ready()
    if not st.get("ok"):
        return {"ok": False, "error": st.get("error") or "worker 未就绪"}
    tmpdir = tempfile.mkdtemp(prefix="fireredasr_")
    wav = os.path.join(tmpdir, "in.wav")
    try:
        if not _to_wav(media, wav):
            return {"ok": False, "error": "抽音轨失败（可能没有音轨）：%s"
                                         % os.path.basename(media)}
        with _LOCK:                      # 一条管道，串起来用
            if _PROC is None or _PROC.poll() is not None:
                return {"ok": False, "error": "worker 已退出"}
            _PROC.stdin.write(json.dumps({"id": req_id, "wav": wav}) + "\n")
            _PROC.stdin.flush()
            line = _readline(CALL_TIMEOUT)
        if not line:
            return {"ok": False, "error": "识别 %.0fs 超时" % CALL_TIMEOUT}
        return json.loads(line)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": "识别失败：%s" % str(exc)[:200]}
    finally:
        for p in (wav,):
            if os.path.isfile(p):
                os.remove(p)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)


def stop() -> None:
    """关掉 worker（任务跑完释放显存；不调也没事，进程退出时一起走）。"""
    global _PROC, _READY
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            try:
                _PROC.stdin.write('{"cmd": "bye"}\n')
                _PROC.stdin.flush()
                _PROC.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                _PROC.kill()
        _PROC, _READY = None, None
