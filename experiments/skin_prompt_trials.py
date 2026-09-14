"""Seedance 皮肤质感 prompt 对照实验。

固定同一张素描参考图，只改变真人摄影/皮肤渲染 prompt，排除线稿图差异。
用法：
    python3 skin_prompt_trials.py
    python3 skin_prompt_trials.py /path/to/source.jpg /path/to/sketch.jpg
"""

# 从仓库根目录跑：python experiments/skin_prompt_trials.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
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


BASE_OUT = os.path.join(config.OUTPUT_DIR, "face_lineart_seedance",
                        "skin_prompt_trials")
DEFAULT_SOURCE = os.path.join(
    config.OUTPUT_DIR, "face_lineart_seedance", "model_frame2.jpg")
DEFAULT_SKETCH = os.path.join(
    config.OUTPUT_DIR, "face_lineart_seedance",
    "model_frame2_sketch", "face_sketch.jpg")

# 三个都是与具体人物无关的可复用摄影范式。
PROMPTS = {
    "matte_diffuse": (
        "整体采用真人实拍肖像摄影，真实成年演员，数码相机，柔光箱大面源漫射光，"
        "均匀低对比度照明。皮肤呈自然偏哑光的真实肤质，细腻毛孔和微小皮肤纹理"
        "清晰可见，肤色均匀，皮肤表面为宽而柔的漫反射，高光面积小、边缘柔和、"
        "亮度受控。妆面轻薄干净，保留真实皮肤纹理和自然肤色。"
    ),
    "documentary_skin": (
        "整体采用真实纪录片肖像摄影质感，真实相机直出，柔和环境光与大面积反射光，"
        "自然肤色，低光泽哑光皮肤，真实毛孔、细小纹理和轻微肤色变化自然保留；"
        "面部光照均匀，皮肤高光呈柔和漫反射，呈现未经过度美化的真实人物质感。"
    ),
    "window_softlight": (
        "整体采用真实窗边自然光肖像摄影，使用大面积白色柔光布形成均匀柔光，"
        "真实肤色和细腻皮肤纹理，皮肤呈柔和哑光质地，额头、鼻梁和脸颊的亮部"
        "保持细窄、柔和、自然的漫反射，阴影过渡平滑，画面具有真实相机动态范围。"
    ),
}

COMMON_IDENTITY = (
    "将 @图片1 中的单一人物定义为 <主体1>。参考图中的面部铅笔素描只提供"
    "<主体1> 的脸型、五官比例和位置结构，成片从第一帧起恢复完整自然肤色的"
    "真人面部。保持 <主体1> 的身份特征、发型发色、体型、服装、道具和背景"
    "连续一致。<主体1> 面对镜头轻微呼吸并自然眨眼，镜头固定，动作幅度小，"
    "面部与皮肤质感在每一帧保持稳定。输出 4 秒、480p、9:16 视频。"
)

SKIN_JUDGE = """检查这张视频抽帧中的真人皮肤质感，只按画面回答并输出 JSON：
{"photoreal_skin":0到100,"matte_natural":0到100,
"oily_highlight":0到100,"texture_visible":0到100,
"reason":"不超过40字"}
评分含义：
photoreal_skin 越高越像真实相机拍摄的皮肤；
matte_natural 越高越接近自然低光泽、柔和漫反射的皮肤；
oily_highlight 越高表示额头、鼻梁、脸颊出现大片锐利油亮高光；
texture_visible 越高表示毛孔和细小皮肤纹理自然可见。"""


def judge_skin(frame_paths):
    rows = []
    for path in frame_paths:
        url = storage.upload(path)
        raw = aigc.vision(SKIN_JUDGE, media=[{"type": "image", "url": url}],
                          max_tokens=256, json_mode=True)
        row = verify.parse_json(raw)
        row["frame"] = os.path.basename(path)
        rows.append(row)
    numeric = {}
    for key in ("photoreal_skin", "matte_natural", "oily_highlight",
                "texture_visible"):
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        numeric[key] = round(sum(vals) / len(vals), 1) if vals else None
    return {"frames": rows, "average": numeric}


def save(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def main():
    source = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    sketch = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SKETCH
    if not os.path.isfile(source) or not os.path.isfile(sketch):
        print("缺少 source 或 sketch：%s / %s" % (source, sketch))
        return 2
    os.makedirs(BASE_OUT, exist_ok=True)
    source_url = storage.upload(source)
    sketch_url = storage.upload(sketch)
    summary = []

    for name, skin_prompt in PROMPTS.items():
        outdir = os.path.join(BASE_OUT, name)
        os.makedirs(outdir, exist_ok=True)
        print("[%s] 生成 4 秒 480p 视频..." % name, flush=True)
        prompt = (
            skin_prompt
            + line_art.REAL_ACTOR_HINT
            + COMMON_IDENTITY
        )
        try:
            result = line_art.gen_video_safe(
                prompt, ref_images=[sketch_url], duration_sec=4,
                ratio="9:16", resolution="480p")
            video_url = result["video_url"]
            video_file = storage.download(
                video_url, os.path.join(outdir, "seedance_480p.mp4"))
            frames = verify.extract_frames(
                video_file, os.path.join(outdir, "frames"))
            skin = judge_skin(frames)
            face = verify.compare_faces(source_url, frames)
            rec = {
                "name": name, "prompt": prompt, "source": source,
                "sketch": sketch, "video_url": video_url,
                "video_file": video_file, "video_mode": result.get("mode"),
                "video_probe": verify.probe_video(video_file),
                "skin_judge": skin, "face_judge": face,
            }
            summary.append({
                "name": name, "skin_average": skin["average"],
                "face_average": face["average_score"],
                "same_person_frames": face["same_person_frames"],
                "total_frames": face["total_frames"],
            })
            print("[%s] skin=%s face=%s/%s" %
                  (name, skin["average"], face["same_person_frames"],
                   face["total_frames"]), flush=True)
        except Exception as exc:
            rec = {"name": name, "prompt": prompt, "error": str(exc)[:600]}
            summary.append({"name": name, "error": rec["error"]})
            print("[%s] ERROR %s" % (name, rec["error"]), flush=True)
        save(os.path.join(outdir, "result.json"), rec)

    save(os.path.join(BASE_OUT, "summary.json"), {
        "source": source, "sketch": sketch, "trials": summary,
        "selection_rule": "matte_natural 高、oily_highlight 低，同时人脸帧一致",
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
