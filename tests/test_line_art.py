"""v2opt24 线稿 prompt 复测：10 张输入图 → 面部线稿 → seedance 视频。

跑法：python3 test_line_art.py
- 输入复用 output/line_art_eval/inputs/p01..p10.jpg
- v2opt24 的 prompt 只在本脚本内临时覆盖 line_art 的模块常量，不改 line_art.py
- 产物落 output/line_art_eval/v2opt24_retest/：线稿图、视频、result.json
"""

# 从仓库根目录跑：python tests/test_line_art.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import concurrent.futures as cf
import json
import os

import aigc
import config
import eval_line_art
import line_art
import storage

TAG = "v2opt24_retest"
OUT = os.path.join(config.OUTPUT_DIR, "line_art_eval", TAG)
VIDEO_PROMPT = "画面中的人物微笑并轻轻转头看向镜头，自然呼吸，镜头缓慢推近，写实电影感"

V2OPT24_REDRAW_BASE = (
    "写实彩色人像摄影原图直出，完整保留头发丝缕、衣物织物、耳颈肌肤及背景的光影色彩与纹理细节，"
    "画面整体为相机直出质感。"
)
V2OPT24_FACE_LINE_ART = (
    "面部正前方受强光过曝褪色，仅在发际线至下颌缘围合的裸露皮肤区域褪去肤色显现纯白底黑色钢笔线稿；"
    "五官线条如底片显影般浮现，留白边界严格顺应真实脸型解剖轮廓收束，绝不越过鬓角或渗入颈部阴影区；"
    "耳部、颈部及衣领维持原照片写实渲染，两种视觉状态在光影交界处形成锐利光学分界。"
)


def run_one(args):
    idx, src = args
    name = "p%02d" % (idx + 1)
    rec = {"name": name, "input": os.path.basename(src)}
    try:
        art = line_art.to_line_art(src)
        rec["line_art_url"] = art
        storage.download(art, os.path.join(OUT, name + "_lineart.jpg"))
        rec.update(eval_line_art.judge(art))
        print("  %s lineart %s" % (name, "PASS" if rec.get("passed") else
                                   "FAIL " + rec.get("reason", "")[:60]))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)[:400]
        rec["passed"] = False
        print("  %s lineart ERROR %s" % (name, str(exc)[:120]))
        return rec

    try:
        vid = aigc.gen_video(VIDEO_PROMPT, ref_images=[rec["line_art_url"]], duration_sec=4)
        rec["video_url"] = vid
        rec["video_ok"] = True
        storage.download(vid, os.path.join(OUT, name + "_video.mp4"))
        print("  %s video OK" % name)
    except Exception as exc:  # noqa: BLE001
        rec["video_ok"] = False
        rec["video_error"] = str(exc)[:400]
        rec["video_real_person_reject"] = line_art.is_real_person_reject(exc)
        print("  %s video ERROR %s" % (name, str(exc)[:120]))
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    line_art._REDRAW_BASE = V2OPT24_REDRAW_BASE
    line_art.FACE_LINE_ART = V2OPT24_FACE_LINE_ART

    srcs = eval_line_art.ensure_inputs()
    print("[%s] 跑 %d 张（线稿 + 视频）..." % (TAG, len(srcs)))
    with cf.ThreadPoolExecutor(3) as ex:
        results = list(ex.map(run_one, list(enumerate(srcs))))

    rec = {"round": TAG,
           "prompt": {"redraw_base": V2OPT24_REDRAW_BASE,
                      "face_line_art": V2OPT24_FACE_LINE_ART},
           "video_prompt": VIDEO_PROMPT,
           "lineart_passed": sum(1 for r in results if r.get("passed")),
           "video_passed": sum(1 for r in results if r.get("video_ok")),
           "total": len(results),
           "fail_counts": {k: sum(1 for r in results if not r.get(k, True))
                           for k in eval_line_art.CHECKS},
           "results": results}
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)

    print("\n线稿通过 %d/%d，视频生成成功 %d/%d"
          % (rec["lineart_passed"], rec["total"], rec["video_passed"], rec["total"]))
    print("各项失败数:", rec["fail_counts"])
    print("产物:", OUT)


if __name__ == "__main__":
    main()
