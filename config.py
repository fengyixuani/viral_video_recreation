"""ViralForge 配置：加载同目录 config.env 并暴露常量（已 export 的环境变量优先）。"""
import os

_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.env")
if os.path.isfile(_CFG):
    with open(_CFG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _env(key: str, default: str) -> str:
    """读环境变量，空值当没配。

    os.getenv(k, default) 只在变量**不存在**时给默认值，显式的空串会被原样返回。
    而 config.env 里 `KEY=` 这种空值行会被上面的加载器照样灌进环境（已有 VF_GPU= 的写法），
    于是 `export GEMINI_GATEWAY_URL=` 或 `BOS_URL_EXPIRE=` 会让本模块 import 就崩
    （IndexError / ValueError），而它被几乎每个模块 import——server、pipeline、CLI 全起不来，
    报错还跟真实原因无关。
    """
    return (os.getenv(key) or "").strip() or default


def _env_int(key: str, default: int) -> int:
    """读整数环境变量，空值或写成非数字时退回默认值（不让 import 崩在这里）。"""
    try:
        return int(_env(key, str(default)))
    except ValueError:
        print("[config] %s 不是整数，按默认值 %d 处理" % (key, default), flush=True)
        return default


# wenchain 网关
BASE_URL = _env("WENCHAIN_BASE_URL", "http://wenku-openai.baidu-int.com/wenchain/strategy").rstrip("/")
API_KEY = _env("WENCHAIN_API_KEY", "")

# 模型
TEXT_MODEL = _env("TEXT_LLM_MODEL", "ali-qwen3.7-max")
VISION_MODEL = _env("VISION_LLM_MODEL", "ali-qwen3.7-plus")
T2I_MODEL = _env("AIGC_T2I_MODEL", "doubao-seedream-5-0-260128")
I2V_MODEL = _env("AIGC_I2V_MODEL", "doubao-seedance-2-0")

# 生成规格
T2I_SIZE = _env("AIGC_T2I_SIZE", "1664x2368")  # 需 ≥3686400 像素，否则 42000002
ASPECT = _env("AIGC_ASPECT", "9:16")
# 音色探针：只为拿一条人声，画质无所谓，用最短时长 + 低分辨率换速度和成本。
# 分辨率值不被网关接受时探针会失败，自动回退到「拿真片段提音色」的老路，不影响出片。
VOICE_PROBE_RESOLUTION = _env("AIGC_VOICE_PROBE_RESOLUTION", "480p")
VOICE_PROBE_SEC = _env_int("AIGC_VOICE_PROBE_SEC", 4)

# 理解类调用（剧本拆解/看图/提示词优化）默认链路：gemini | qwen
UNDERSTAND_ENGINE = _env("AIGC_ENGINE", "gemini")

# Gemini 视觉理解（wenchain 内网 openairr 网关 + Gemini 原生 contents.parts 协议，媒体 base64 内联）
# 该网关只有内网 IP、没有域名，节点会漂移，所以配成列表逐个试；GEMINI_GATEWAY_URL 可用逗号分隔覆盖。
GEMINI_DEFAULT_GATEWAYS = (
    "http://10.252.161.175:8018,"
    "http://10.252.161.216:8534,http://10.252.161.216:8535,"
    "http://10.252.164.45:2540,http://10.252.164.176:8049,http://10.252.164.237:2076,"
    "http://10.252.165.27:2056,http://10.252.165.158:2044,http://10.252.165.158:8113"
)
GEMINI_GATEWAYS = [u.strip().rstrip("/") for u
                   in _env("GEMINI_GATEWAY_URL", GEMINI_DEFAULT_GATEWAYS).split(",") if u.strip()]
if not GEMINI_GATEWAYS:      # 只写了逗号这种，退回内置列表，别让 [0] 抛 IndexError
    GEMINI_GATEWAYS = [u.strip().rstrip("/") for u in GEMINI_DEFAULT_GATEWAYS.split(",")]
GEMINI_GATEWAY = GEMINI_GATEWAYS[0]
GEMINI_ENDPOINT = _env("GEMINI_GATEWAY_ENDPOINT", "/service/llm/openairr")
GEMINI_CHANNEL = _env("GEMINI_CHANNEL", "") or API_KEY
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.1-pro-preview")
# 10 段在 no_proxy 里，默认会绕过代理导致无路由，所以这条链路显式指定代理走出去。
GEMINI_PROXY = _env("GEMINI_PROXY", "") or (os.getenv("http_proxy") or "").strip()
GEMINI_TIMEOUT = _env_int("GEMINI_TIMEOUT", 300)
GEMINI_RETRIES = max(1, _env_int("GEMINI_RETRIES", 3))

# BOS
BOS_AK = os.getenv("BOS_ACCESS_KEY_ID", "")
BOS_SK = os.getenv("BOS_SECRET_ACCESS_KEY", "")
BOS_BUCKET = _env("BOS_BUCKET", "netdisk-scan-internal")
BOS_HOST = _env("BOS_HOST", "bj.bcebos.com")
BOS_PREFIX = _env("BOS_KEY_PREFIX", "viralforge").strip("/")
BOS_EXPIRE = _env_int("BOS_URL_EXPIRE", -1)  # -1 = 长期有效

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
