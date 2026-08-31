# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""文件存储：BOS 上传（本地文件 → 公网预签名 URL）与结果下载。

网关和视频生成模型只能拉公网 URL，本地图/视频/音频都得先 upload()。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import hashlib
import os

import requests

from . import config

_client = None


def _bos():
    """惰性建 BOS 客户端：没配密钥的场景（只跑本地剪辑）不该因为导入就报错。"""
    global _client
    if _client is None:
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient
        if not (config.BOS_AK and config.BOS_SK):
            raise RuntimeError("请先配置 BOS_ACCESS_KEY_ID / BOS_SECRET_ACCESS_KEY")
        _client = BosClient(BceClientConfiguration(
            credentials=BceCredentials(config.BOS_AK, config.BOS_SK),
            endpoint=config.BOS_HOST))
    return _client


def fingerprint(path: str, size: int) -> str:
    """内容指纹：文件大小 + 首尾各 1MB 的 md5（大文件不整读）。"""
    md5 = hashlib.md5(str(size).encode())
    with open(path, "rb") as fh:
        md5.update(fh.read(1 << 20))
        if size > (1 << 20):
            fh.seek(size - (1 << 20))
            md5.update(fh.read(1 << 20))
    return md5.hexdigest()


def object_key(path: str) -> str:
    """object key 用内容指纹，同内容重复上传覆盖同一个 key。"""
    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower() or ".bin"
    return "%s/%s%s" % (config.BOS_PREFIX, fingerprint(path, size), ext)


def upload(path: str) -> str:
    """上传本地文件到 BOS，返回预签名 HTTPS URL（GET 有效，HEAD 会 403）。"""
    from baidubce import protocol
    from baidubce.http import http_methods

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError("文件不存在：%s" % path)
    key = object_key(path)
    try:
        _bos().put_object_from_file(config.BOS_BUCKET, key, path)
    except Exception:   # noqa: BLE001  SDK 不走代理，直连失败时改用预签名 PUT
        put_url = _bos().generate_pre_signed_url(
            config.BOS_BUCKET, key, expiration_in_seconds=1800,
            protocol=protocol.HTTPS, httpmethod=http_methods.PUT).decode()
        with open(path, "rb") as src:
            resp = requests.put(put_url, data=src, timeout=600)
        if resp.status_code not in (200, 201):
            raise RuntimeError("BOS 上传失败 HTTP %s: %s"
                               % (resp.status_code, resp.text[:300]))
    return _bos().generate_pre_signed_url(
        config.BOS_BUCKET, key, expiration_in_seconds=config.BOS_EXPIRE,
        protocol=protocol.HTTPS).decode()


def download(url: str, out_path: str) -> str:
    """把生成结果（图/视频公网 URL）下载到本地。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                fh.write(chunk)
    return out_path
