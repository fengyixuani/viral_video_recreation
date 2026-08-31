# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2026 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""配置层：加载 conf/config.env 并暴露常量。

优先级：进程环境变量 > conf/config.local.env（本地私密覆盖，不入库） > conf/config.env。
所以调参改文件、临时改用 export，密钥只放 local 或环境变量里。

Authors: fangmuyuan(fangmuyuan@baidu.com)
Date:    2026/08/31
"""

import os

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


def _load_env_file(path: str) -> None:
    """把 KEY=VALUE 读进 os.environ，已存在的键不覆盖（先加载的优先）。"""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# VVR_CONFIG_ENV 可整体换一份配置；local 先于默认加载，因此 local 里的值生效
_load_env_file(os.getenv("VVR_CONFIG_ENV", "").strip())
_load_env_file(os.path.join(CONF_DIR, "config.local.env"))
_load_env_file(os.path.join(CONF_DIR, "config.env"))

# ---------------- wenchain 网关 ----------------
BASE_URL = os.getenv("WENCHAIN_BASE_URL",
                     "http://wenku-openai.baidu-int.com/wenchain/strategy").rstrip("/")
API_KEY = os.getenv("WENCHAIN_API_KEY", "")

# ---------------- 模型 ----------------
TEXT_MODEL = os.getenv("TEXT_LLM_MODEL", "ali-qwen3.7-max")
VISION_MODEL = os.getenv("VISION_LLM_MODEL", "ali-qwen3.7-plus")
T2I_MODEL = os.getenv("AIGC_T2I_MODEL", "doubao-seedream-5-0-260128")
I2V_MODEL = os.getenv("AIGC_I2V_MODEL", "doubao-seedance-2-0")

# ---------------- 生成规格 ----------------
T2I_SIZE = os.getenv("AIGC_T2I_SIZE", "1664x2368")   # 需 ≥3686400 像素，否则报 42000002
ASPECT = os.getenv("AIGC_ASPECT", "9:16")

# 理解类调用默认链路：gemini | qwen（gemini 不通时自动回落 wenchain）
UNDERSTAND_ENGINE = os.getenv("AIGC_ENGINE", "gemini").strip()

# ---------------- Gemini 视觉理解网关 ----------------
# 该网关只有内网 IP、没有域名，节点会漂移，所以配成列表逐个试
GEMINI_GATEWAYS = [u.strip().rstrip("/")
                   for u in os.getenv("GEMINI_GATEWAY_URL", "").split(",") if u.strip()]
GEMINI_ENDPOINT = os.getenv("GEMINI_GATEWAY_ENDPOINT", "/service/llm/openairr")
GEMINI_CHANNEL = os.getenv("GEMINI_CHANNEL") or API_KEY
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
# 10 段在 no_proxy 里，默认会绕过代理导致无路由，所以这条链路显式指定代理走出去
GEMINI_PROXY = os.getenv("GEMINI_PROXY", os.getenv("http_proxy", "")).strip()
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "300"))
GEMINI_RETRIES = max(1, int(os.getenv("GEMINI_RETRIES", "3")))

# ---------------- BOS 对象存储 ----------------
BOS_AK = os.getenv("BOS_ACCESS_KEY_ID", "")
BOS_SK = os.getenv("BOS_SECRET_ACCESS_KEY", "")
BOS_BUCKET = os.getenv("BOS_BUCKET", "")
BOS_HOST = os.getenv("BOS_HOST", "bj.bcebos.com")
BOS_PREFIX = os.getenv("BOS_KEY_PREFIX", "viral_video_recreation").strip("/")
BOS_EXPIRE = int(os.getenv("BOS_URL_EXPIRE", "-1"))   # -1 = 长期有效

# 产物根目录：默认落在当前工作目录下的 output/，不写进安装目录
OUTPUT_DIR = os.path.abspath(os.getenv("VVR_OUTPUT_DIR",
                                       os.path.join(os.getcwd(), "output")))


def missing_keys() -> "list[str]":
    """返回缺失的必填配置名，供 cmdline doctor 自检用。"""
    need = {"WENCHAIN_API_KEY": API_KEY, "BOS_ACCESS_KEY_ID": BOS_AK,
            "BOS_SECRET_ACCESS_KEY": BOS_SK, "BOS_BUCKET": BOS_BUCKET}
    return [k for k, v in need.items() if not v]
