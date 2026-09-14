"""无五官人物图 + 独立五官图 -> Seedance -> 人脸一致性校验。

这条链路把身份信息拆成两个互不混淆的参考素材：

1. featureless_person.jpg：保留头发、身体、服装、饰品和背景，只把眉眼鼻口
   编辑成连续的真实肤色无五官面；
2. isolated_face_features.jpg：纯白背景，只保留同一人物的眉毛、双眼、鼻子和嘴巴。

Seedance 阶段严格按 [无五官人物图, 白底五官图] 的顺序传两张 reference_image，
固定生成 4 秒、480p、9:16 视频，然后抽 6 帧与原始真人照片比较。

用法：
    python3 face_feature_split_seedance.py
    python3 face_feature_split_seedance.py /path/to/source.jpg
    python3 face_feature_split_seedance.py /path/to/source.jpg output_tag

本脚本不使用 argparse。
"""

# 从仓库根目录跑：python experiments/face_feature_split_seedance.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
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
DEFAULT_TAG = "feature_split_trial"
MAX_IMAGE_ATTEMPTS = 3

# 两类都是与具体人物无关的通用模板；输入人物的外观只来自 ref_images。
FACELESS_PROMPTS = [
    line_art.FACELESS_PERSON_EDIT_PROMPT,
    line_art.FACELESS_PERSON_EDIT_PROMPT_ALT,
    line_art.FACELESS_PERSON_EDIT_PROMPT_CLEAN,
]
FEATURE_PROMPTS = [
    line_art.ISOLATED_FACE_FEATURES_PROMPT,
    line_art.ISOLATED_FACE_FEATURES_PROMPT_ALT,
    line_art.ISOLATED_FACE_FEATURES_PROMPT_CLEAN,
]

VIDEO_PROMPT = line_art.SPLIT_FACE_VIDEO_HINT

FEATURELESS_JUDGE_PROMPT = """检查这张人物参考图是否是一张“无五官人物外观图”，严格只输出 JSON：
{"no_eyes":true或false,"no_nose":true或false,"no_mouth":true或false,
"no_eyebrows":true或false,"face_shape_kept":true或false,
"hair_preserved":true或false,"body_clothes_preserved":true或false,
"background_ok":true或false,"photoreal":true或false,
"reason":"不超过50字"}
判定标准：脸部仍有自然的脸型、肤色和皮肤体积，但眉毛、眼睛、睫毛、虹膜、瞳孔、
鼻梁、鼻翼、鼻孔、嘴唇、唇缝和牙齿都已经由连续皮肤覆盖；头发、耳朵、颈部、
身体、服装和饰品保持真实照片质感；背景可以延续原图，也可以是干净的中性纯色背景，
但不能出现手、家具、文字等与人物外观无关的杂物。"""

FEATURES_JUDGE_PROMPT = """检查这张图片是否是一张“纯白背景孤立五官参考图”，严格只输出 JSON：
{"white_background":true或false,"only_features":true或false,
"eyebrows_present":true或false,"eyes_present":true或false,
"nose_present":true或false,"mouth_present":true或false,
"layout_clear":true或false,"no_full_face":true或false,
"no_hair":true或false,"no_body":true或false,
"reason":"不超过50字"}
判定标准：画面只出现同一人物的两条眉毛、双眼、鼻子和嘴巴，按眉毛在上、
眼睛在眉毛下、鼻子居中、嘴巴在下的相对关系清晰排列，并保留输入图的朝向和透视；
背景必须是干净纯白，
不得出现完整脸、脸部轮廓、皮肤面、头发、耳朵、颈部、身体、服装、场景或文字。"""

FACELESS_CHECKS = (
    "no_eyes", "no_nose", "no_mouth", "no_eyebrows", "face_shape_kept",
    "hair_preserved", "body_clothes_preserved", "background_ok",
    "photoreal",
)
FEATURE_CHECKS = (
    "white_background", "only_features", "eyebrows_present", "eyes_present",
    "nose_present", "mouth_present", "layout_clear", "no_full_face",
    "no_hair", "no_body",
)


def parse_json(text):
    return verify.parse_json(text)


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "是", "通过")
    return bool(value)


def judge_image(url, prompt, checks):
    raw = aigc.vision(prompt, media=[{"type": "image", "url": url}],
                      max_tokens=512, json_mode=True)
    item = parse_json(raw)
    item["passed"] = all(as_bool(item.get(key)) for key in checks)
    item["score"] = sum(as_bool(item.get(key)) for key in checks)
    return item


def _candidate_score(item):
    judge = item.get("judge") or {}
    try:
        return int(judge.get("score", 0))
    except (TypeError, ValueError):
        return 0


def generate_candidates(kind, source, prompts, judge_prompt, checks, outdir):
    """生成并质检一类参考图，失败时换通用模板，返回最佳候选。"""
    candidates = []
    for attempt in range(1, MAX_IMAGE_ATTEMPTS + 1):
        prompt = prompts[(attempt - 1) % len(prompts)]
        rec = {"kind": kind, "attempt": attempt, "prompt": prompt}
        attempt_dir = os.path.join(outdir, "%s_attempt_%02d" % (kind, attempt))
        os.makedirs(attempt_dir, exist_ok=True)
        try:
            print("  [%s %d/%d] 生成参考图..." %
                  (kind, attempt, MAX_IMAGE_ATTEMPTS), flush=True)
            url = aigc.gen_image(prompt, ref_images=[source])
            path = storage.download(
                url, os.path.join(attempt_dir, "%s.jpg" % kind))
            rec.update({"url": url, "file": path})
            try:
                rec["judge"] = judge_image(url, judge_prompt, checks)
            except Exception as exc:  # VLM 失败不丢掉已经生成的图片
                rec["judge_error"] = str(exc)[:600]
                rec["judge"] = {"passed": False, "score": 0,
                                "reason": "质检调用失败"}
            candidates.append(rec)
            print("    质检：%s（%d/%d）" %
                  ("PASS" if rec["judge"].get("passed") else "FAIL",
                   rec["judge"].get("score", 0), len(checks)), flush=True)
            if rec["judge"].get("passed"):
                return rec, candidates
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)[:600]
            candidates.append(rec)
            print("    生成失败：%s" % rec["error"], flush=True)
    if not candidates:
        raise RuntimeError("%s 没有生成候选图" % kind)
    # 即使视觉质检没有全通过，也选信息保留最多的一张继续做一次视频实测。
    valid = [item for item in candidates if item.get("url")]
    if not valid:
        errors = "; ".join(item.get("error", "未知错误") for item in candidates)
        raise RuntimeError("%s 全部生成失败：%s" % (kind, errors[:800]))
    return max(valid, key=_candidate_score), candidates


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


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

    result = {"source": source, "source_url": source_url}
    print("[1/4] 生成人物外观参考图和独立五官参考图...", flush=True)
    try:
        featureless, featureless_trials = generate_candidates(
            "featureless_person", source, FACELESS_PROMPTS,
            FEATURELESS_JUDGE_PROMPT, FACELESS_CHECKS, outdir)
        features, feature_trials = generate_candidates(
            "isolated_face_features", source, FEATURE_PROMPTS,
            FEATURES_JUDGE_PROMPT, FEATURE_CHECKS, outdir)
    except Exception as exc:  # noqa: BLE001
        result.update({
            "status": "image_error",
            "image_error": str(exc)[:800],
        })
        result_path = os.path.join(outdir, "result.json")
        save_json(result_path, result)
        print("[4/4] 参考图阶段失败，结果已保存：%s" % result_path, flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 1

    # 统一落一份易找的最终素材，实际传给 Seedance 的 URL 与这里一致。
    featureless_file = storage.download(
        featureless["url"], os.path.join(outdir, "featureless_person.jpg"))
    features_file = storage.download(
        features["url"], os.path.join(outdir, "isolated_face_features.jpg"))
    featureless["selected_file"] = featureless_file
    features["selected_file"] = features_file

    print("[2/4] 用两张参考图调用 Seedance 4 秒 480p...", flush=True)
    # 参考图绑定和任务描述放在前面，真人摄影/皮肤质感放在其后，避免白底图被当成场景。
    prompt = VIDEO_PROMPT + line_art.REAL_ACTOR_PHOTO_HINT
    result.update({
        "featureless_person": featureless,
        "isolated_face_features": features,
        "reference_quality_passed": (
            as_bool((featureless.get("judge") or {}).get("passed"))
            and as_bool((features.get("judge") or {}).get("passed"))
        ),
        "image_trials": {
            "featureless_person": featureless_trials,
            "isolated_face_features": feature_trials,
        },
        "video_prompt": prompt,
        "duration_sec": 4,
        "resolution": "480p",
        "ratio": "9:16",
        "reference_count": 2,
        "requested_reference_order": [
            "featureless_person (@图片1)",
            "isolated_face_features (@图片2)",
        ],
    })
    try:
        generated = line_art.gen_video_safe(
            prompt,
            ref_images=[featureless["url"], features["url"]],
            duration_sec=4,
            ratio="9:16",
            resolution="480p",
        )
        video_url = generated["video_url"]
        video_file = storage.download(
            video_url, os.path.join(outdir, "seedance_480p.mp4"))
        result.update({
            "video_url": video_url,
            "video_file": video_file,
            "video_mode": generated.get("mode"),
            "used_reference_urls": generated.get("ref_images"),
        })

        print("[3/4] 抽取视频帧并校验人脸一致性...", flush=True)
        frames = verify.extract_frames(video_file, os.path.join(outdir, "frames"))
        result["video_probe"] = verify.probe_video(video_file)
        result["face_judge"] = verify.compare_faces(source_url, frames)
        result["status"] = (
            "passed" if result["face_judge"].get("passed") else "face_failed")
    except Exception as exc:  # noqa: BLE001
        result["status"] = "video_error"
        result["video_error"] = str(exc)[:800]
        result["video_real_person_reject"] = line_art.is_real_person_reject(exc)

    result["passed"] = result.get("status") == "passed"
    result_path = os.path.join(outdir, "result.json")
    save_json(result_path, result)
    print("[4/4] 结果已保存：%s" % result_path, flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
