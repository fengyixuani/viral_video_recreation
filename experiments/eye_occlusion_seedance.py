"""只遮挡双眼的真人参考图 -> Seedance -> 6 帧一致性验证。

这是一条最小对照链路：

1. 从输入照片复制全部像素，只在双眼位置增加眼部遮挡；
2. 将这一张图直接作为 Seedance 的唯一 reference_image；
3. 固定 4 秒、480p、9:16，抽 6 帧并与原始真人照片比较。

用法：
    python3 eye_occlusion_seedance.py
    python3 eye_occlusion_seedance.py /path/to/source.jpg
    python3 eye_occlusion_seedance.py /path/to/source.jpg output_tag

本脚本不使用 argparse。
"""

# 从仓库根目录跑：python experiments/eye_occlusion_seedance.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import sys

import aigc
import config
import line_art
import run_face_lineart_trials as verify
import storage


BASE_OUT = os.path.join(config.OUTPUT_DIR, "face_lineart_seedance")
DEFAULT_SOURCE = os.path.join(BASE_OUT, "model_frame2.jpg")
DEFAULT_TAG = "eye_occlusion_trial"
STYLES = ("sunglasses", "eye_band")

VIDEO_PROMPT = (
    "将 @图片1 中的单一人物定义为 <主体1>，只从 @图片1 读取人物的发型发色、"
    "脸型、鼻子、嘴巴、体型、服装、饰品、背景、构图和光线。"
    "参考图中的眼部遮挡物只覆盖双眼，遮挡边界保持稳定，不扩展到鼻子、嘴巴、"
    "脸颊、耳朵、头发或服装；成片保持同一遮挡物的形状和位置。"
    "输出一段4秒、480p、9:16竖屏真人实拍视频，固定近景镜头，单一成年人物面对镜头"
    "自然呼吸、轻微点头并说一句短话，动作幅度小且连续，手部不进入画面。"
    "保持真实相机直出质感、自然肤色、低光泽哑光皮肤、真实毛孔、真实发丝和真实衣物纤维，"
    "鼻子、嘴巴、脸型、发型、服装和身份特征在每一帧保持稳定；无字幕、无水印、无额外人物。"
    "口播台词为：{大家好，今天分享一个简单实用的小技巧。}"
)

OCCLUDED_COMPARE_PROMPT = """第一张图片是原始真人参考图，第二张图片是视频的一帧。
参考图和视频帧中的双眼可能被同一副墨镜或眼部遮挡物覆盖，因此不要把眼睛、眉毛
是否可见作为差异。只比较仍然可见的身份特征：鼻梁和鼻尖轮廓、嘴巴比例、下颌和
脸型轮廓、发际线、发型、耳饰和整体相貌。忽略表情、角度、光线、画质、遮挡物、
服装和背景差异。严格只输出 JSON：
{"same_person":true或false,"score":0到100的数字,"reason":"不超过40字"}"""


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def parse_json(text):
    text = (text or "").strip().strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM 未返回 JSON：%s" % text[:240])
    return json.loads(text[start:end + 1])


def compare_occluded_faces(source_url, frame_paths):
    results = []
    for path in frame_paths:
        frame_url = storage.upload(path)
        raw = aigc.vision(
            OCCLUDED_COMPARE_PROMPT,
            media=[
                {"type": "image", "url": source_url},
                {"type": "image", "url": frame_url},
            ],
            max_tokens=256,
            json_mode=True,
        )
        item = parse_json(raw)
        item["frame"] = os.path.basename(path)
        results.append(item)
        print("      %s: score=%s same=%s" %
              (item["frame"], item.get("score"), item.get("same_person")),
              flush=True)
    scores = []
    for item in results:
        try:
            scores.append(float(item.get("score")))
        except (TypeError, ValueError):
            pass
    same = sum(bool(item.get("same_person")) for item in results)
    average = sum(scores) / len(scores) if scores else 0.0
    return {
        "frames": results,
        "average_score": round(average, 1),
        "same_person_frames": same,
        "total_frames": len(results),
        "passed": bool(results) and same / len(results) >= 0.75
                   and average >= 70,
    }


def run_variant(source, source_url, outdir, style):
    variant_dir = os.path.join(outdir, style)
    os.makedirs(variant_dir, exist_ok=True)
    rec = {
        "style": style,
        "source": source,
        "prompt": VIDEO_PROMPT,
        "duration_sec": 4,
        "resolution": "480p",
        "ratio": "9:16",
        "reference_count": 1,
    }
    try:
        reference_file = line_art.make_eye_occluded_person(
            source,
            os.path.join(variant_dir, "eye_occluded_reference.jpg"),
            style=style,
        )
        reference_url = storage.upload(reference_file)
        rec.update({
            "reference_file": reference_file,
            "reference_url": reference_url,
        })
        print("  [%s] 直接调用 Seedance..." % style, flush=True)
        # 不调用 gen_video_safe，确保结果准确回答“仅遮眼参考图
        # 是否能以 ref2v 进入 Seedance”，不会被其它参考图替代。
        video_url = aigc.gen_video(
            VIDEO_PROMPT,
            ref_images=[reference_url],
            duration_sec=4,
            ratio="9:16",
            resolution="480p",
        )
        video_file = storage.download(
            video_url, os.path.join(variant_dir, "seedance_480p.mp4"))
        frames = verify.extract_frames(
            video_file, os.path.join(variant_dir, "frames"))
        probe = verify.probe_video(video_file)
        face_judge = compare_occluded_faces(source_url, frames)
        rec.update({
            "video_url": video_url,
            "video_file": video_file,
            "video_mode": "ref2v",
            "seedance_ref2v_passed": True,
            "used_exact_reference": True,
            "used_reference_urls": [reference_url],
            "video_probe": probe,
            "audio_track_present": "Audio:" in (probe.get("probe") or ""),
            "face_judge": face_judge,
            "status": "passed" if face_judge.get("passed") else "face_failed",
        })
    except Exception as exc:  # noqa: BLE001
        rec.update({
            "status": "seedance_rejected_or_error",
            "error": str(exc)[:1000],
            "real_person_reject": line_art.is_real_person_reject(exc),
        })
        print("  [%s] 失败：%s" % (style, rec["error"]), flush=True)
    save_json(os.path.join(variant_dir, "result.json"), rec)
    return rec


def main():
    source = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    tag = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG
    if not os.path.isfile(source):
        print("输入图片不存在：%s" % source)
        return 2
    if not tag or any(c not in
                      "abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                      for c in tag):
        print("输出目录名只能包含字母、数字、连字符和下划线")
        return 2

    outdir = os.path.join(BASE_OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    source_url = storage.upload(source)
    print("输入：%s" % source, flush=True)
    results = []
    for style in STYLES:
        print("[%d/%d] 测试 %s..." %
              (len(results) + 1, len(STYLES), style), flush=True)
        rec = run_variant(source, source_url, outdir, style)
        results.append(rec)
        if rec.get("status") == "passed":
            break

    summary = {
        "source": source,
        "source_url": source_url,
        "video_prompt": VIDEO_PROMPT,
        "attempted_styles": [item["style"] for item in results],
        "results": results,
        "seedance_ref2v_passed": any(
            item.get("seedance_ref2v_passed") for item in results),
        "passed": any(item.get("status") == "passed" for item in results),
    }
    save_json(os.path.join(outdir, "result.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
