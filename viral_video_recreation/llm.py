# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""模型调用层：LLM 文本 / VLM 视觉理解 / 文生图 / 图生视频。

全部走 wenchain 内网网关，两种鉴权：
- LLM / VLM：OpenAI 兼容 /chat/completions，Authorization: Bearer <WENCHAIN_API_KEY>
- 生图 / 生视频：/incommonuserr，鉴权靠 payload 里的 channel（= WENCHAIN_API_KEY）

另有一条并行的视觉链路 vision_gemini()：走独立的 Gemini 网关，Gemini 原生
contents.parts 协议，媒体 base64 内联，支持音频。wenchain 的 /chat/completions
不认 gemini 模型名，所以必须分成两个函数。

本模块只做模型调用，文件上传下载见 storage.py。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import base64
import json
import mimetypes
import os
import random
import time
from typing import Any, Optional

import requests

from . import config

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")


# ---------------- OpenAI 兼容：LLM / VLM ----------------
def _chat_completions(messages: "list[dict[str, Any]]", model: str, max_tokens: int,
                      json_mode: bool) -> str:
    """OpenAI 兼容接口的统一发送，返回第一条 choice 的文本。"""
    payload: "dict[str, Any]" = {"model": model, "temperature": 0.2,
                                 "max_tokens": max_tokens, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(
        config.BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer %s" % config.API_KEY,
                 "Content-Type": "application/json"},
        json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError("chat HTTP %s: %s" % (resp.status_code, resp.text[:400]))
    return resp.json()["choices"][0]["message"]["content"]


def chat(user: str, system: str = "You are a helpful assistant.",
         model: Optional[str] = None, max_tokens: int = 4096,
         json_mode: bool = False) -> str:
    """纯文本对话，返回回复文本。json_mode=True 时 prompt 里必须出现 json 字样。"""
    return _chat_completions(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model or config.TEXT_MODEL, max_tokens, json_mode)


def vision(user: str, media: "Optional[list[dict[str, Any]]]" = None,
           system: str = "You are a helpful assistant.", model: Optional[str] = None,
           max_tokens: int = 8192, json_mode: bool = False) -> str:
    """VLM 视觉理解（看图/看视频），返回回复文本。

    media 为 [{"type": "video"|"image", "url": 公网URL}]；本地文件先 storage.upload()。
    """
    blocks: "list[dict[str, Any]]" = [{"type": "text", "text": user}]
    for item in media or []:
        kind = item.get("type") or "video"
        key = "video_url" if kind == "video" else "image_url"
        blocks.append({"type": key, key: {"url": item["url"]}})
    return _chat_completions(
        [{"role": "system", "content": system}, {"role": "user", "content": blocks}],
        model or config.VISION_MODEL, max_tokens, json_mode)


# ---------------- Gemini 视觉理解（独立网关，与 vision() 并行） ----------------
# 踩坑记录：成功判据是 status.code == 0 而非只看 HTTP 200；理解类请求不要传
# generationConfig，thinkingConfig.thinkingLevel=minimal 会让 gemini 报 INVALID_ARGUMENT。
def _gemini_part(item: Any) -> "dict[str, Any]":
    """把一个媒体转成 inlineData part。item 可为本地路径、公网 URL，或 {"url"/"path", "type"}。"""
    if isinstance(item, dict):
        uri = str(item.get("url") or item.get("path") or "")
        kind = str(item.get("type") or "")
    else:
        uri, kind = str(item), ""
    clean = uri.split("?")[0]
    if uri.startswith(("http://", "https://")):
        resp = requests.get(uri, timeout=300)
        resp.raise_for_status()
        raw = resp.content
    else:
        with open(uri, "rb") as fh:
            raw = fh.read()
    mime = mimetypes.guess_type(clean)[0]
    if not mime:
        ext = os.path.splitext(clean)[1].lower()
        if ext in VIDEO_EXTS or kind == "video":
            mime = "video/mp4"
        elif ext in AUDIO_EXTS or kind == "audio":
            mime = "audio/mp3"
        else:
            mime = "image/png"
    return {"inlineData": {"mimeType": mime,
                           "data": base64.b64encode(raw).decode("ascii")}}


def _gemini_text(body: "dict[str, Any]") -> str:
    """从 Gemini 响应里拼出文本，结构不对就返回空串交给上层重试。"""
    try:
        parts: Any = body["data"]["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(parts, list):
        return ""
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


def _gemini_proxies():
    """gemini 网关是内网 IP，落在 no_proxy 的 10.0.0.0/8 里，必须显式指定代理才有路由。"""
    proxy = config.GEMINI_PROXY
    return {"http": proxy, "https": proxy} if proxy else None


def vision_gemini(user: str, media: "Optional[list[Any]]" = None, system: str = "",
                  model: Optional[str] = None, timeout: Optional[int] = None,
                  retries: Optional[int] = None) -> str:
    """Gemini 多模态理解（图 / 视频 / 音频 + 文本），返回回复文本。

    media 元素可为本地路径、公网 URL，或 {"type": "video"|"image"|"audio", "url": ...}；
    一律读成 base64 内联，不走 URL 透传，因此不需要先 storage.upload()。
    system 拼在提问前（Gemini 原生协议没有 system 角色）；要 json 就在 user 里直接要求，
    本链路没有 response_format。全部重试失败抛 RuntimeError。
    """
    if not config.GEMINI_GATEWAYS:
        raise RuntimeError("未配置 GEMINI_GATEWAY_URL")
    text = (system.strip() + "\n\n" + user) if system.strip() else user
    parts: "list[dict[str, Any]]" = [{"text": text}]
    parts += [_gemini_part(m) for m in media or [] if m]
    retries = retries or config.GEMINI_RETRIES
    last = "unknown"
    for attempt in range(retries):
        for gateway in config.GEMINI_GATEWAYS:
            stamp = int(time.time() * 1000)
            body: "dict[str, Any]" = {
                "api_name": config.GEMINI_ENDPOINT, "channel": config.GEMINI_CHANNEL,
                "chat_id": stamp, "query_id": stamp, "stream": False,
                "user": "viral_video_recreation", "model": model or config.GEMINI_MODEL,
                "contents": {"parts": parts, "role": "user"}}
            try:
                resp = requests.post(gateway + config.GEMINI_ENDPOINT,
                                     headers={"Content-Type": "application/json"},
                                     json=body, timeout=timeout or config.GEMINI_TIMEOUT,
                                     proxies=_gemini_proxies())
            except (requests.RequestException, OSError) as exc:
                last = "%s 请求失败: %s" % (gateway, exc)
                continue
            if resp.status_code != 200:
                last = "%s HTTP %s: %s" % (gateway, resp.status_code, resp.text[:200])
                continue
            data: "dict[str, Any]" = resp.json()
            status: "dict[str, Any]" = data.get("status") or {}
            if status.get("code") != 0:
                last = "%s 网关错误 code=%s msg=%s" % (gateway, status.get("code"),
                                                      status.get("msg"))
                continue
            out = _gemini_text(data)
            if out:
                return out
            last = "%s 返回无文本（可能被截断）" % gateway
        if attempt + 1 < retries:
            time.sleep(min(8.0, 2.0 * (2 ** attempt)))
    raise RuntimeError("gemini 调用失败: %s" % last)


def understand(user: str, media: "Optional[list[dict[str, Any]]]" = None,
               system: str = "You are a helpful assistant.", max_tokens: int = 8192,
               json_mode: bool = False, engine: Optional[str] = None) -> str:
    """理解类调用的统一入口：默认 gemini，gemini 不通时回落 wenchain。

    engine 默认取 config.UNDERSTAND_ENGINE（环境变量 AIGC_ENGINE）。回落时有 media
    走 vision()，纯文本走 chat()；两条链路入参语义一致，调用方不关心用的是哪个模型。
    gemini 没有 response_format，需要 json 就在 prompt 里写清楚（json_mode 只对回落生效）。
    """
    engine = engine or config.UNDERSTAND_ENGINE
    if engine == "gemini":
        try:
            return vision_gemini(user, media=media, system=system)
        except (RuntimeError, OSError) as exc:
            print("gemini 不可用，回落 wenchain：%s" % str(exc)[:160], flush=True)
    if media:
        return vision(user, media=media, system=system,
                      max_tokens=max_tokens, json_mode=json_mode)
    return chat(user, system=system, max_tokens=max_tokens, json_mode=json_mode)


# ---------------- /incommonuserr 网关（图 / 视频生成） ----------------
def _post(model: str, tag: str, options: "dict[str, Any]", timeout: int) -> "dict[str, Any]":
    """按 /incommonuserr 协议发一次生成请求，成功判据是 status.code == 0。"""
    qid = int(time.time() * 1000) + random.randint(0, 999)
    msgs = [{"role": "user", "content": tag}]
    payload: "dict[str, Any]" = {
        "channel": config.API_KEY, "chat_id": qid, "query_id": qid, "model": model,
        "stream": False, "message_user": msgs, "messages": msgs, "message_prompt": msgs}
    payload.update(options)
    resp = requests.post(config.BASE_URL + "/incommonuserr",
                         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                         headers={"Content-Type": "application/json"}, timeout=timeout)
    body: "dict[str, Any]" = resp.json()
    if (body.get("status") or {}).get("code") != 0:
        raise RuntimeError("wenchain 生成失败: %s"
                           % json.dumps(body, ensure_ascii=False)[:500])
    return body


def gen_image(prompt: str, size: Optional[str] = None,
              ref_images: "Optional[list[str]]" = None) -> str:
    """文/图生图，返回图片 URL。ref_images 可为本地路径（转 base64）或公网 URL。"""
    opts: "dict[str, Any]" = {"prompt": prompt, "size": size or config.T2I_SIZE, "n": 1,
                              "response_format": "url", "watermark": False}
    refs: "list[str]" = []
    for ref in ref_images or []:
        if os.path.isfile(ref):
            ext = (os.path.splitext(ref)[1].lstrip(".").lower() or "jpeg")
            ext = ext.replace("jpg", "jpeg")
            with open(ref, "rb") as fh:
                refs.append("data:image/%s;base64,%s"
                            % (ext, base64.b64encode(fh.read()).decode()))
        else:
            refs.append(ref)
    if refs:
        opts["image"] = refs
    body = _post(config.T2I_MODEL, "t2i", {"seedream_options": opts}, 300)
    for item in ((body.get("data") or {}).get("data") or []):
        url = item.get("bos_url") or item.get("url")
        if url:
            return url
    raise RuntimeError("生图响应缺少图片 url: %s"
                       % json.dumps(body, ensure_ascii=False)[:400])


def gen_video(prompt: str, first_frame_url: Optional[str] = None,
              last_frame_url: Optional[str] = None,
              ref_images: "Optional[list[str]]" = None,
              ref_videos: "Optional[list[str]]" = None,
              ref_audios: "Optional[list[str]]" = None,
              duration_sec: int = 5, ratio: Optional[str] = None,
              resolution: Optional[str] = None) -> str:
    """视频生成，返回视频 URL。三种图生模式互斥：

    - 文生视频：只给 prompt
    - 首帧 / 首尾帧：first_frame_url（+ last_frame_url）
    - 多模态参考生视频：ref_images(≤9) / ref_videos(≤3) / ref_audios(≤3)，
      至少要有 1 图或 1 视频，音频不能单独用。

    素材必须是公网可拉取 URL（本地文件先 storage.upload()）。
    含真人人脸的参考图会被风控拒（40000002）。
    """
    dur = max(4, min(15, int(duration_sec)))
    options = "%s  --ratio %s  --dur %d" % (prompt.strip(), ratio or config.ASPECT, dur)
    if resolution:
        options += "  --resolution %s" % resolution
    content: "list[dict[str, Any]]" = [{"type": "text", "text": options}]
    if first_frame_url:
        content.append({"type": "image_url", "image_url": {"url": first_frame_url},
                        "role": "first_frame"})
        if last_frame_url:
            content.append({"type": "image_url", "image_url": {"url": last_frame_url},
                            "role": "last_frame"})
    else:
        for url in ref_images or []:
            content.append({"type": "image_url", "image_url": {"url": url},
                            "role": "reference_image"})
        for url in ref_videos or []:
            content.append({"type": "video_url", "video_url": {"url": url},
                            "role": "reference_video"})
        for url in ref_audios or []:
            content.append({"type": "audio_url", "audio_url": {"url": url},
                            "role": "reference_audio"})
    body = _post(config.I2V_MODEL, "i2v" if len(content) > 1 else "t2v",
                 {"seedancepro_options": {"content": content}}, 900)
    data: "dict[str, Any]" = body.get("data") or {}
    node = data.get("content")
    url: str = node.get("video_url", "") if isinstance(node, dict) else ""
    if not url and data.get("result"):
        result: "dict[str, Any]" = json.loads(data["result"]).get("content") or {}
        url = result.get("video_url", "")
    if not url:
        raise RuntimeError("视频生成缺少 video_url: %s"
                           % json.dumps(body, ensure_ascii=False)[:400])
    return url

