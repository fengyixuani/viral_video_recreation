"""最小真人图 -> 脸部线稿 -> Seedance -> 人脸一致性验证链路。

用法：
    python3 face_lineart_seedance.py /path/to/real-person.jpg

输入必须是一张带清晰真人脸的图片。产物写入 output/face_lineart_seedance/：
line_art.jpg、seedance_480p.mp4、frames/*.jpg、report.json。

本脚本不使用 argparse，避免把这条一次性验证链路包装成额外配置系统。
"""

# 从仓库根目录跑：python experiments/face_lineart_seedance.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import subprocess
import sys

import aigc
import line_art
import storage


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output", "face_lineart_seedance")
FRAME_DIR = os.path.join(OUT_DIR, "frames")

VIDEO_PROMPT = (
    "画面采用真人实拍电影摄影质感，真实成年演员，真实肤色、毛孔、细小皮肤纹理、"
    "眼睛湿润反光、独立发丝体积、真实衣物纤维与自然阴影。"
    "将 @图片1 中的单一人物定义为 <主体1>，保持 <主体1> 稳定的身份、脸型、"
    "五官比例、发型发色、体型和服装。"
    "参考图中的白色面部线稿只提供脸型和五官结构，成片恢复为完整自然肤色的人类面部，"
    "线稿介质不参与最终画面。人物面对镜头轻微呼吸并自然眨眼，镜头固定，"
    "<主体1> 面对镜头轻微呼吸并自然眨眼，镜头固定，影棚柔光方向稳定，"
    "动作幅度小且连贯，画面保持单一人物和连续稳定的面部。"
)

COMPARE_PROMPT = """比较两张图片中的人物是否是同一个人。
第一张是原始真人参考图，第二张是视频抽帧。
只比较可见的人脸身份特征：脸型、眼鼻嘴比例、眉形、发际线和整体相貌；
忽略表情、光线、角度、压缩、服装和背景差异。
严格只输出 JSON：{"same_person":true或false,"score":0到100的数字,"reason":"不超过30字"}"""


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def extract_frames(video, count=8):
    os.makedirs(FRAME_DIR, exist_ok=True)
    # 4 秒视频均匀抽 8 帧，首尾也纳入检查。
    paths = []
    for i in range(count):
        sec = 0.05 + (3.90 * i / max(1, count - 1))
        dst = os.path.join(FRAME_DIR, "frame_%02d.jpg" % (i + 1))
        ret = run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", "%.3f" % sec, "-i", video, "-frames:v", "1",
                   "-q:v", "3", dst])
        if ret.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError("抽帧失败：%s" % ret.stderr[-300:])
        paths.append(dst)
    return paths


def parse_json(text):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM 未返回 JSON：%s" % text[:200])
    return json.loads(text[start:end + 1])


def compare_faces(reference_url, frame_paths):
    results = []
    for path in frame_paths:
        frame_url = storage.upload(path)
        raw = aigc.vision(COMPARE_PROMPT, media=[
            {"type": "image", "url": reference_url},
            {"type": "image", "url": frame_url},
        ], max_tokens=256)
        item = parse_json(raw)
        item["frame"] = os.path.relpath(path, OUT_DIR)
        results.append(item)
        print("[face] %s score=%s same=%s" %
              (item["frame"], item.get("score"), item.get("same_person")))
    scores = [float(x["score"]) for x in results if str(x.get("score", "")).strip()]
    same_count = sum(bool(x.get("same_person")) for x in results)
    return {
        "frames": results,
        "average_score": round(sum(scores) / len(scores), 1) if scores else None,
        "same_person_frames": same_count,
        "total_frames": len(results),
        "passed": bool(results) and same_count / len(results) >= 0.75
                   and (sum(scores) / len(scores) >= 70 if scores else False),
    }


def main():
    if len(sys.argv) != 2:
        print("用法：python3 face_lineart_seedance.py /path/to/real-person.jpg")
        return 2
    source = os.path.abspath(sys.argv[1])
    if not os.path.isfile(source):
        print("输入图片不存在：%s" % source)
        return 2
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] 真人图上传...")
    source_url = storage.upload(source)
    print("[2/4] 生成脸部线稿、真实头发与服装参考图...")
    line_art_url = line_art.to_line_art(source)
    line_art_path = storage.download(line_art_url, os.path.join(OUT_DIR, "line_art.jpg"))

    print("[3/4] 调用 Seedance 生成 4 秒 480p 视频...")
    video_result = line_art.gen_video_safe(
        line_art.REAL_ACTOR_HINT + VIDEO_PROMPT,
        ref_images=[line_art_url], duration_sec=4, ratio="9:16",
        resolution="480p")
    video_url = video_result["video_url"]
    final_video = storage.download(video_url, os.path.join(OUT_DIR, "seedance_480p.mp4"))

    print("[4/4] 抽帧并校验人脸一致性...")
    frames = extract_frames(final_video)
    report = compare_faces(source_url, frames)
    report.update({
        "source": source,
        "line_art": line_art_path,
        "video": final_video,
        "video_url": video_url,
        "video_mode": video_result.get("mode"),
        "duration_sec": 4,
        "resolution": "480p",
        "prompt": VIDEO_PROMPT,
    })
    report_path = os.path.join(OUT_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
