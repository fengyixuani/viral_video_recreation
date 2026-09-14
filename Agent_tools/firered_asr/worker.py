"""firered_asr worker —— 常驻 ASR 进程：模型只加载一次，之后按行收请求、按行回结果。

单独起进程有两个原因：几个 G 的权重/引擎不该常驻在 pipeline 进程里；加载要二十几秒，
每块补片各起一次进程等于白等一分钟。所以由 firered_asr.start() 提前拉起它，
generate 步用到时模型已经热了。

两个后端，靠 FIRERED_BACKEND 选（默认 trt）：
- trt   TensorRT 引擎（firered-trt env）。实测热态单块 0.09s，比 torch 快 5~8 倍。
        代价是引擎与 GPU 架构绑定（本机是 L20 + TensorRT 10.10.0.31），换卡要重建。
- torch PyTorch FP32（base conda）。慢一些但不依赖引擎，是引擎失效时的退路。
两者输出同一套 schema（assemble_result 与 fireredasr2system 的字段一致），
差异只有：TRT 的 asr_confidence 恒为 0（TensorRT-LLM 解码器不吐 token 概率），
词级时间戳与 torch 有 ≤120ms 的偏差（实测本机 85ms）。

协议（stdin/stdout 各一行一条 json，避免任何长度前缀协商）：
    <- {"ready": true, "load_sec": 24.2, "backend": "trt"}   启动完成，只发一次
    -> {"id": "seg01_2", "wav": "/tmp/x.wav"}     请求（wav 必须 16k 单声道 PCM16）
    <- {"id": "seg01_2", "ok": true, "text": ..., "sentences": [...], "words": [...]}
    -> {"cmd": "bye"}                             退出
失败也回一行 {"ok": false, "error": ...}，绝不让调用方等超时。
"""
import json
import os
import sys
import time


def _build_torch():
    """PyTorch FP32 后端，返回 run(wav, uttid) -> 原始结果 dict。

    return_timestamp 默认是 False，不显式打开就拿不到词级时间戳（words 会是空表），
    而「台词说到哪」全靠词级时间。enable_lid 必须关：本机没下 LID 权重，
    用上游默认配置会直接起不来。
    """
    repo = os.getenv("FIRERED_REPO", "/home/work/chengzhiyang/asr_test/FireRedASR2S")
    models = os.getenv("FIRERED_MODELS", "/home/work/chengzhiyang/asr_test/models")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredasr2system import (FireRedAsr2System,
                                                FireRedAsr2SystemConfig)
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    gpu = os.getenv("FIRERED_USE_GPU", "1").strip() != "0"
    system = FireRedAsr2System(FireRedAsr2SystemConfig(
        vad_model_dir=os.path.join(models, "FireRedVAD", "VAD"),
        asr_type="aed",
        asr_model_dir=os.path.join(models, "FireRedASR2-AED"),
        punc_model_dir=os.path.join(models, "FireRedPunc"),
        vad_config=FireRedVadConfig(use_gpu=gpu),
        asr_config=FireRedAsr2Config(use_gpu=gpu, use_half=False, beam_size=3,
                                     return_timestamp=True),
        punc_config=FireRedPuncConfig(use_gpu=gpu, sentence_max_length=-1),
        asr_batch_size=7, punc_batch_size=24,
        enable_vad=True, enable_lid=False, enable_punc=True))
    return lambda wav, uttid: system.process(wav, uttid)


def _build_trt():
    """TensorRT 后端，返回 run(wav, uttid) -> 原始结果 dict。

    复用 firered_speed/tensorrt_full_pipeline 的 load_models + run_inference_once：
    引擎只反序列化一次，之后每条请求只跑推理。必须满足三件事，否则起不来：
    - 解释器是 firered-trt env（TensorRT 10.10.0.31 + TensorRT-LLM 0.20.0）；
    - LD_LIBRARY_PATH 含该 env 的 lib（缺了报 libpython3.12.so.1.0 / libmpi.so 找不到），
      这个由客户端 firered_asr.start() 负责注入；
    - device-id 用进程内的逻辑编号：CUDA_VISIBLE_DEVICES 已经把目标卡映射成 0，
      再传物理卡号会报 invalid device ordinal。
    启动时那句「flashinfer is not installed properly」是无害的：本链路用 legacy Session。
    """
    speed = os.getenv("FIRERED_SPEED_REPO", "/home/work/chengzhiyang/firered_speed")
    if speed not in sys.path:
        sys.path.insert(0, speed)
    import torch
    import tensorrt_full_pipeline as tfp
    from pathlib import Path

    args = tfp.parse_args(["--input", os.path.join(speed, "README.md"),
                           "--output-dir", "/tmp/firered_trt_unused"])
    args.device_id = int(os.getenv("FIRERED_DEVICE_ID", "0"))
    torch.cuda.set_device(args.device_id)
    resources = tfp.load_models(args)

    def run(wav, uttid):
        """跑一条：先按 16k/单声道/PCM16 严格校验，再走引擎。"""
        wav_np, sr, dur = tfp.validate_wav(Path(wav))
        args.input = Path(wav)
        result, _events, _timings = tfp.run_inference_once(
            args, resources, wav_np, sr, dur, uttid)
        return result

    return run


def _build():
    """按 FIRERED_BACKEND 装配后端，返回 (run, 后端名)。"""
    backend = (os.getenv("FIRERED_BACKEND") or "trt").strip().lower()
    if backend == "torch":
        return _build_torch(), "torch"
    return _build_trt(), "trt"


def _reply(obj):
    """回一行 json 并立刻 flush（不 flush 会让调用方一直阻塞在 readline）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _one(run, req):
    """跑一条 ASR，把时间戳统一换成秒（调用方只关心秒，别让它再去除 1000）。"""
    wav = req.get("wav") or ""
    if not os.path.isfile(wav):
        return {"ok": False, "error": "wav 不存在：%s" % wav}
    t0 = time.time()
    r = run(wav, req.get("id") or "req")
    return {
        "ok": True,
        "text": r.get("text") or "",
        "dur_s": round(float(r.get("dur_s") or 0), 3),
        "asr_sec": round(time.time() - t0, 2),
        "sentences": [{"start": round(s["start_ms"] / 1000.0, 3),
                       "end": round(s["end_ms"] / 1000.0, 3),
                       "text": s.get("text") or "",
                       "confidence": round(float(s.get("asr_confidence") or 0), 3)}
                      for s in (r.get("sentences") or [])],
        "words": [{"start": round(w["start_ms"] / 1000.0, 3),
                   "end": round(w["end_ms"] / 1000.0, 3),
                   "text": w.get("text") or ""}
                  for w in (r.get("words") or [])],
        "vad": [[round(a / 1000.0, 3), round(b / 1000.0, 3)]
                for a, b in (r.get("vad_segments_ms") or [])],
    }


def main():
    """加载模型 → 报 ready → 循环收请求。任何单条请求的异常都只回错误，不退出。"""
    # FireRed 各模块往 stdout 打 INFO 日志，会污染协议。先把 stdout 换成 stderr，
    # 只有 _reply 用真正的 stdout。
    real_out, sys.stdout = sys.stdout, sys.stderr
    t0 = time.time()
    try:
        run, backend = _build()
    except Exception as exc:  # noqa: BLE001
        sys.stdout = real_out
        _reply({"ready": False, "error": "模型加载失败：%s" % str(exc)[:300]})
        return 1
    load = time.time() - t0
    sys.stdout = real_out
    _reply({"ready": True, "load_sec": round(load, 2), "backend": backend})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            _reply({"ok": False, "error": "请求不是合法 json：%s" % exc})
            continue
        if req.get("cmd") == "bye":
            return 0
        sys.stdout = sys.stderr          # 推理期间的 INFO 日志同样挡掉
        try:
            out = _one(run, req)
        except Exception as exc:  # noqa: BLE001
            out = {"ok": False, "error": "识别失败：%s" % str(exc)[:300]}
        sys.stdout = real_out
        out["id"] = req.get("id") or ""
        _reply(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
