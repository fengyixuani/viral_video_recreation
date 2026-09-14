"""最小“脸部素描 -> Seedance -> 抽帧人脸校验”实验。

用法：
    python3 face_sketch_seedance.py
    python3 face_sketch_seedance.py /path/to/source.jpg

默认输入 output/face_lineart_seedance/source.jpg。
"""

# 从仓库根目录跑：python experiments/face_sketch_seedance.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import sys

import config
import line_art
import run_face_lineart_trials as verify
import storage


DEFAULT_SOURCE = os.path.join(
    config.OUTPUT_DIR, "face_lineart_seedance", "source.jpg")
BASE_OUT_DIR = os.path.join(config.OUTPUT_DIR, "face_lineart_seedance")
OUT_DIR = os.path.join(BASE_OUT_DIR, "sketch_trial")

VIDEO_PROMPT = (
    "输出一段4秒、480p、竖屏视频。将 @图片1 中的单一人物定义为 <主体1>。"
    "面部铅笔素描只提供 <主体1> 的脸型、五官比例和位置结构，成片从第一帧起恢复"
    "完整自然肤色的真人面部。保持 <主体1> 的身份特征、发型发色、体型、服装、"
    "道具和背景连续一致。<主体1> 面对镜头轻微呼吸并自然眨眼，镜头固定，"
    "影棚柔光稳定，动作幅度小，人物面部清晰且连续稳定。"
)


def main():
    global OUT_DIR
    source = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not os.path.isfile(source):
        print("输入图片不存在：%s" % source)
        return 2
    tag = sys.argv[2] if len(sys.argv) > 2 else "sketch_trial"
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                      for c in tag):
        print("输出目录名只能包含字母、数字、连字符和下划线")
        return 2
    OUT_DIR = os.path.join(BASE_OUT_DIR, tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    source_url = storage.upload(source)

    print("[1/3] 图生图：仅把脸部改为铅笔素描...", flush=True)
    sketch_url = line_art.to_face_sketch(source)
    sketch_file = storage.download(
        sketch_url, os.path.join(OUT_DIR, "face_sketch.jpg"))

    print("[2/3] Seedance 生成 4 秒 480p 真人视频...", flush=True)
    prompt = line_art.REAL_ACTOR_HINT + VIDEO_PROMPT
    generated = line_art.gen_video_safe(
        prompt, ref_images=[sketch_url], duration_sec=4,
        ratio="9:16", resolution="480p")
    video_url = generated["video_url"]
    video_file = storage.download(
        video_url, os.path.join(OUT_DIR, "seedance_480p.mp4"))

    print("[3/3] 抽帧校验人脸一致性...", flush=True)
    frame_dir = os.path.join(OUT_DIR, "frames")
    frames = verify.extract_frames(video_file, frame_dir)
    face_judge = verify.compare_faces(source_url, frames)
    report = {
        "source": source,
        "sketch_prompt": line_art.FACE_SKETCH_EDIT_PROMPT,
        "sketch_url": sketch_url,
        "sketch_file": sketch_file,
        "video_prompt": prompt,
        "video_url": video_url,
        "video_file": video_file,
        "video_mode": generated.get("mode"),
        "video_probe": verify.probe_video(video_file),
        "face_judge": face_judge,
        "passed": face_judge["passed"],
    }
    verify.save_json(os.path.join(OUT_DIR, "result.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
