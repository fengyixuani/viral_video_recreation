"""模型/存储连通性测试：python3 test_aigc.py [llm|vlm|image|video|bos|all]"""

# 从仓库根目录跑：python tests/test_aigc.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import aigc
import config
import storage

TMP = "/tmp/viralforge_probe.jpg"


def t_llm():
    print("[llm]", aigc.TEXT_MODEL)
    print("  ->", aigc.chat("用一句话说明什么是短视频爆款钩子。")[:200])


def t_image():
    print("[image]", aigc.T2I_MODEL)
    url = aigc.gen_image("竖屏电商图：一瓶精华液放在白色大理石台面上，柔光，高级感")
    print("  ->", url[:100])
    return url


def t_video():
    print("[video]", aigc.I2V_MODEL)
    print("  ->", aigc.gen_video("镜头缓慢推近一瓶精华液，柔光，高级感", duration_sec=4)[:100])


def t_bos():
    print("[bos]", config.BOS_BUCKET)
    storage.download(t_image(), TMP)
    url = storage.upload(TMP)
    print("  ->", url[:100])
    code = __import__("requests").get(url, timeout=60, stream=True).status_code
    os.remove(TMP)
    if code != 200:
        raise RuntimeError("预签名 URL 不可读 HTTP %s" % code)


def t_vlm():
    """VLM 要公网 URL，所以先生成一张图，让模型描述它。"""
    print("[vlm]", aigc.VISION_MODEL)
    url = t_image()
    print("  ->", aigc.vision("这张图里有什么？一句话描述。",
                              media=[{"type": "image", "url": url}])[:200])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {"llm": [t_llm], "vlm": [t_vlm], "image": [t_image], "video": [t_video],
             "bos": [t_bos], "all": [t_llm, t_vlm, t_image, t_video, t_bos]}[which]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("  PASS\n")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("  FAIL: %s\n" % exc)
    sys.exit(1 if failed else 0)
