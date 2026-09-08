"""ViralForge 文件存储：BOS 上传（本地文件 → 公网预签名 URL）+ 结果下载。

网关和 seedance 只能拉公网 URL，本地图/视频/音频都得先 upload()。
"""
import hashlib
import os

import requests

import config

_client = None


def _bos():
    global _client
    if _client is None:
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient
        if not (config.BOS_AK and config.BOS_SK):
            raise RuntimeError("请先配置 BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY")
        _client = BosClient(BceClientConfiguration(
            credentials=BceCredentials(config.BOS_AK, config.BOS_SK), endpoint=config.BOS_HOST))
    return _client


def _fingerprint(path: str, size: int) -> str:
    """内容指纹：文件大小 + 首尾各 1MB 的 md5（大文件不整读）。"""
    h = hashlib.md5(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (1 << 20):
            fh.seek(size - (1 << 20))
            h.update(fh.read(1 << 20))
    return h.hexdigest()


def upload(path: str) -> str:
    """上传本地文件到 BOS，返回预签名 HTTPS URL（GET 有效，HEAD 会 403）。

    object key 用内容指纹，同内容重复上传覆盖同一个 key。
    """
    from baidubce import protocol
    from baidubce.http import http_methods

    path = os.path.abspath(path)
    size = os.path.getsize(path)
    key = "%s/%s%s" % (config.BOS_PREFIX, _fingerprint(path, size),
                       os.path.splitext(path)[1].lower() or ".bin")
    try:
        _bos().put_object_from_file(config.BOS_BUCKET, key, path)
    except Exception:  # SDK 不走代理，直连失败时改用预签名 PUT
        put_url = _bos().generate_pre_signed_url(
            config.BOS_BUCKET, key, expiration_in_seconds=1800,
            protocol=protocol.HTTPS, httpmethod=http_methods.PUT).decode()
        with open(path, "rb") as src:
            resp = requests.put(put_url, data=src, timeout=600)
        if resp.status_code not in (200, 201):
            raise RuntimeError("BOS 上传失败 HTTP %s: %s" % (resp.status_code, resp.text[:300]))
    return _bos().generate_pre_signed_url(
        config.BOS_BUCKET, key, expiration_in_seconds=config.BOS_EXPIRE,
        protocol=protocol.HTTPS).decode()


def download(url: str, out_path: str) -> str:
    """把生成结果（图/视频公网 URL）下载到本地。

    落盘后要校验：网关限流或 CDN 出错时会返回 HTTP 200 + 一段 HTML 错误页，
    只 raise_for_status 的话它会被原样写成 *.jpg / *.mp4，然后作为「图片」发给 VLM 比对，
    比对失败又被判成「判不出」，在 unknown_ok 的链路上当成通过写进事实卡。
    宁可在这里报错，让调用方的重试/降级看得见。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and not (ctype.startswith(("image/", "video/", "audio/"))
                          or ctype == "application/octet-stream"):
            raise RuntimeError("下载内容不是媒体（Content-Type: %s）：%s" % (ctype, url[:120]))
        got = 0
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
                got += len(chunk)
    if got < 1024:      # 正经的图/视频不会只有几百字节，多半是错误页或空 body
        raise RuntimeError("下载内容只有 %d 字节，判为无效产物：%s" % (got, url[:120]))
    return out_path
