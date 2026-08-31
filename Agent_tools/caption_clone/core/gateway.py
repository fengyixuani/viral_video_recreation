"""gateway —— 替代原 whq_clone/pipeline_utils + as_core：LLM/VLM 网关调用与阶段缓存。

原包的 LLM 调用走 Agent 的 `pipeline_utils.ask_qianfan`（内部再复用 `as_core` 的网关配置），
VLM 模型名取 `as_core.VISION_MODEL`。抽成独立工具后这里用 requests 直连同一个 wenchain 网关，
不再依赖 Agent 的 src/shared，配置全部走环境变量。

外部依赖：`requests`（必需）、`json_repair`（可选，JSON 兜底修复）。
"""
import hashlib
import json
import os
import re
import time

import requests

BASE_URL = os.getenv("WENCHAIN_BASE_URL", "http://wenku-openai.baidu-int.com/wenchain/strategy")
# 网关的 key 其实是租户标识，沿用源项目 as_core 的默认值；换租户用 WENCHAIN_API_KEY 覆盖。
API_KEY = os.getenv("WENCHAIN_API_KEY", os.getenv("QIANFAN_API_KEY", "wangpantob_all_video_copy"))
# 拆块/纠错/挑花字用文本模型；读参考帧字幕用视觉模型
TEXT_MODEL = os.getenv("TEXT_LLM_MODEL", os.getenv("LLM_MODEL", "ali-qwen3.7-max"))
VISION_MODEL = os.getenv("VISION_LLM_MODEL", "ali-qwen3.7-plus")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))

try:
    import json_repair as _json_repair
except ImportError:  # pragma: no cover
    _json_repair = None


def ask_qianfan(messages, model=None, max_tokens=None, temperature=0.2, timeout=None):
    """调 wenchain 的 chat/completions，返回 ``(content_text, full_json)``。

    强制 ``response_format=json_object``（本工具所有调用都要 JSON）。网络类错误重试
    QIANFAN_RETRY_COUNT 次（默认 3），HTTP 4xx 之类不可重试的直接抛。
    """
    timeout = timeout or int(os.getenv("LLM_TIMEOUT", "300"))
    if not API_KEY:
        raise RuntimeError("WENCHAIN_API_KEY 未配置（字幕克隆要调 LLM/VLM 网关）")
    payload = {
        "model": model or TEXT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.8,
        "max_tokens": max_tokens or MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    attempts = max(1, int(os.getenv("QIANFAN_RETRY_COUNT", "3")))
    retry_sleep = float(os.getenv("QIANFAN_RETRY_SLEEP", "2"))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                BASE_URL.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer {}".format(API_KEY),
                         "Content-Type": "application/json"},
                json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], data
        except (requests.HTTPError, requests.RequestException) as exc:
            body = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            status = exc.response.status_code if getattr(exc, "response", None) is not None else "?"
            last_error = RuntimeError("HTTP {}: {}".format(status, body[:400]))
            retryable = (isinstance(exc, requests.RequestException)
                         and not isinstance(exc, requests.HTTPError)) or any(
                tok in body.lower() for tok in
                ("connection reset", "conn talk failed", "llm_rr_error"))
            if attempt < attempts and retryable:
                print("[gateway] 重试 {}/{}: {}".format(attempt, attempts,
                                                       str(last_error)[-200:]), flush=True)
                time.sleep(retry_sleep)
                continue
            raise last_error from exc
    raise last_error or RuntimeError("ask_qianfan failed without a captured error")


def strip_json_noise(text):
    """去掉模型输出外层的 ```json 围栏 / BOM 等噪声。"""
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.replace("\ufeff", "").strip()


def extract_json_text(text):
    """从模型输出里截出最外层 {...} 的 JSON 文本；截不到抛 ValueError。"""
    if "## 模型输出" in text:
        text = text.split("## 模型输出", 1)[1]
    text = strip_json_noise(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object")
    return text[start:end + 1]


def loads_with_repair(text):
    """解析 JSON，失败时用 json_repair 兜底；两者都失败才抛原始错误。"""
    raw = extract_json_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if _json_repair is None:
            raise
        try:
            return _json_repair.loads(raw)
        except Exception:
            raise exc


def parallel_map(fn, items, workers=None, env_var="WHQ_LLM_CONCURRENCY", default=3):
    """按序返回 [fn(x) for x in items]，但并发执行（LLM/VLM 是网络等待型）。

    fn 抛异常时该项返回异常对象，由调用方降级 —— 单点失败不阻断主流程。
    """
    items = list(items)
    if not items:
        return []
    if workers is None:
        workers = int(os.getenv(env_var, str(default)))
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        out = []
        for x in items:
            try:
                out.append(fn(x))
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
        return out
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, x) for x in items]
        results = []
        for fu in futures:
            try:
                results.append(fu.result())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
    return results


# ---------- 阶段缓存（参考风格分析很贵，同一条参考视频只跑一次） ----------

def fingerprint(*parts):
    """把若干可 JSON 化的部分拼成 16 位 sha1 指纹（阶段缓存的 key）。"""
    h = hashlib.sha1()
    for p in parts:
        try:
            h.update(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode())
        except (TypeError, ValueError):
            h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def file_fingerprint(paths):
    """文件列表的 (路径, 大小, mtime)：输入换了就让缓存失效。"""
    out = []
    for p in paths or []:
        try:
            st = os.stat(p)
            out.append([os.path.abspath(p), st.st_size, int(st.st_mtime)])
        except OSError:
            out.append([str(p), -1, -1])
    out.sort()
    return out


def load_stage(path, fp):
    """读阶段缓存；指纹不符/文件缺失/CAPTION_RESUME=0 时返回 None（需要重跑）。"""
    if os.getenv("CAPTION_RESUME", os.getenv("WHQ_RESUME", "1")) in ("0", "false", "False"):
        return None
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if obj.get("_fingerprint") == fp else None


def save_stage(path, fp, payload):
    """把 payload 带上指纹落盘为阶段缓存；写失败只告警不抛。"""
    obj = dict(payload)
    obj["_fingerprint"] = fp
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        print("[gateway] 阶段缓存写入失败(忽略): {}".format(str(exc)[:120]), flush=True)
    return path
