"""真人照片的线稿参考图 + Seedance 人脸一致性多轮验证。

用法：
    python3 run_face_lineart_trials.py
    python3 run_face_lineart_trials.py /path/to/source.jpg

默认读取 output/face_lineart_seedance/source.jpg，最多尝试 100 轮。
每轮都使用与具体人物无关的 prompt 模板；人物服装等描述由 VLM 自动填入
模板变量，不写死在 prompt 里。找到同时满足线稿和视频人脸一致性的结果后停止。
100 轮仍失败时，自动进入“脸部线稿 + 独立服装参考图”备用方案。

本脚本不使用 argparse。
"""

# 从仓库根目录跑：python experiments/run_face_lineart_trials.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import subprocess
import sys

import aigc
import config
import line_art
import storage


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(
    config.OUTPUT_DIR, "face_lineart_seedance", "source.jpg")
TRIAL_ROOT = os.path.join(config.OUTPUT_DIR, "face_lineart_seedance", "trials")
MAX_ROUNDS = 100
FRAME_COUNT = 6

# 这些模板只描述“材质边界”和“生成任务”，不绑定任何具体人物。
# desc 会单独作为自动提取的外观槽位写入 prompt。
BASE_PHOTO = (
    "真人实拍人物摄影，数码单反相机，85mm定焦镜头，f/2.8光圈，ISO 200，"
    "1/160s快门，影棚柔光箱三点布光，实体照片扫描质感，真实光学成像；"
    "真实头发丝、真实肤色与毛孔、真实眼睛反光、真实衣物纤维和自然阴影。"
)
FACE_TEMPLATES = [
    (
        "局部材质替换：仅将眉眼鼻嘴所在的裸露面部区域变为纯白底黑色细线钢笔线稿，"
        "线稿边界严格沿真实脸型从发际线到下颌线闭合；头发、耳朵、颈部、肩膀和服装保持照片质感。"
    ),
    (
        "面部作为独立平面：面部皮肤区域只保留白底黑线勾勒的眉眼鼻嘴和脸部轮廓，"
        "边缘贴合脸型，不越过发际线、耳朵和下颌；头发、裸露身体和全部衣物仍是彩色实拍材质。"
    ),
    (
        "做精确的面部拼贴效果：五官区域呈白色纸面与黑色细线结构，"
        "只覆盖脸部，不覆盖发丝、耳部、脖子或衣服；其余所有区域维持真实相机照片的颜色、纹理和光影。"
    ),
    (
        "将脸部当作受控的线稿遮罩：遮罩必须是随脸型变化的自然椭圆轮廓，"
        "内部是眉眼鼻嘴的黑色线条和白色留白；遮罩外的发型、皮肤、手臂、背景与服装全部写实保留。"
    ),
    (
        "真实照片中的局部线描：仅把面部裸露皮肤去色成白底黑色连续线稿，"
        "保留可辨认的脸型与五官比例；任何头发、衣领、服装、手和背景都不得线稿化，继续呈现真实材质。"
    ),
    (
        "建立清晰的材质分界：脸部五官为极简黑色钢笔轮廓与白色留白，"
        "分界沿发际线、脸颊和下巴自然收束；脸外的头发和服装保持真实色彩、褶皱、纤维与高光。"
    ),
    (
        "人像照片局部风格编辑：只编辑正面面部区域为干净的白底黑线五官线稿，"
        "不改变头发长度和发色，不改变服装款式和颜色，不把线条扩散到任何面部之外的区域。"
    ),
    (
        "面部线稿占位参考图：线稿只作为脸型和五官的结构占位，"
        "它必须完整贴合脸部轮廓并与真实头发、颈部、身体和服装形成锐利自然的材质边界。"
    ),
]
PHASE_MODIFIERS = [
    "边界优先，线条细而连续，留白均匀。",
    "优先保持原始构图、人物比例与服装细节。",
    "优先保持头发和衣物的真实色彩与纹理。",
    "优先保证面部线稿闭合且不出现矩形切边。",
    "优先保证五官位置、大小和脸型比例稳定。",
]

VIDEO_PROMPT = (
    "输出一段4秒、480p、竖屏视频。画面从第一帧起保持真人实拍电影摄影质感，"
    "呈现真实成年演员、自然肤色、毛孔和细小皮肤纹理，真实眼睛湿润反光，"
    "真实发丝体积与光泽，真实衣物纤维、褶皱和阴影。"
    "将 @图片1 中的单一人物定义为 <主体1>，保持稳定的身份特征、脸型、"
    "五官比例、发型发色、体型和服装。参考图中的白色面部线稿只提供脸型和五官"
    "结构，成片恢复为完整自然肤色的人类面部，线稿介质不参与最终画面。"
    "<主体1> 面对镜头轻微呼吸并自然眨眼，镜头固定，影棚柔光方向稳定，"
    "动作幅度小且连贯，画面保持单一人物、干净背景、连续稳定的面部与服装。"
)

LINE_JUDGE_PROMPT = """严格检查这张参考图是否满足“只有脸部是线稿、其余保持真实照片”。
输出 JSON，不要输出解释文字：
{"face_line_art":true或false,"hair_photo":true或false,
"clothes_photo":true或false,"skin_photo":true或false,
"background_photo":true或false,"boundary_fits":true或false,
"reason":"不超过40字"}
判定标准：面部眉眼鼻嘴和脸型是白底黑线；头发、耳朵、颈部、手、衣服、背景
必须保留真实颜色、纹理和光影；面部白色区域必须沿脸型自然收束，不能是矩形、
圆角矩形或越界遮罩。"""

FACE_COMPARE_PROMPT = """第一张图片是原始真人参考图，第二张图片是生成视频的一帧。
判断两张图里可见的人是否为同一个人，只比较身份特征：脸型、眼睛、眉形、
鼻子、嘴部比例、发际线和整体相貌。忽略表情、角度、光线、画质、服装和背景。
严格只输出 JSON：
{"same_person":true或false,"score":0到100的数字,"reason":"不超过40字"}"""

FALLBACK_FACE_PROMPT = (
    "通用人物脸部结构参考图：正面单人头肩构图，脸部占画面主体，"
    "面部是白底黑色细线钢笔线稿，只表现脸型、眉眼鼻嘴和五官比例，"
    "不出现真实肤色和阴影；头发只保留简洁真实的发型轮廓，背景干净，"
    "这是一张供视频模型恢复真人身份的结构参考图。"
)
FALLBACK_CLOTHES_PROMPT = (
    "通用人物服装参考图：保留参考人物的真实发型、体型和服装，"
    "展示从肩部到膝部的真实摄影质感，完整保留衣物颜色、款式、材质、"
    "褶皱和光影；头部不画清晰五官，使用背面或无五官轮廓，"
    "画面只强调服装和身体比例，纯色背景，无文字。"
)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def parse_json(text):
    text = (text or "").strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM 未返回 JSON：%s" % text[:240])
    return json.loads(text[start:end + 1])


def describe_once(source):
    """只读一次人物外观；描述是变量，不属于针对某张图写死的模板。"""
    return line_art.describe_person(source).strip()


def line_prompt(desc, round_no):
    template = FACE_TEMPLATES[(round_no - 1) % len(FACE_TEMPLATES)]
    modifier = PHASE_MODIFIERS[((round_no - 1) // len(FACE_TEMPLATES))
                               % len(PHASE_MODIFIERS)]
    return "%s%s人物外观参考：%s。%s" % (
        BASE_PHOTO, template, desc, modifier)


def judge_line_art(url):
    raw = aigc.vision(LINE_JUDGE_PROMPT, media=[{"type": "image", "url": url}],
                      max_tokens=512, json_mode=True)
    obj = parse_json(raw)
    checks = ("face_line_art", "hair_photo", "clothes_photo",
              "skin_photo", "background_photo", "boundary_fits")
    obj["passed"] = all(bool(obj.get(k)) for k in checks)
    return obj


def extract_frames(video, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i in range(FRAME_COUNT):
        # 不取最后一帧，避免恰好落在编码尾部黑帧。
        sec = 0.08 + (3.78 * i / max(1, FRAME_COUNT - 1))
        dst = os.path.join(outdir, "frame_%02d.jpg" % (i + 1))
        ret = run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", "%.3f" % sec, "-i", video, "-frames:v", "1",
                   "-q:v", "3", dst])
        if ret.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError("抽帧失败：%s" % ret.stderr[-300:])
        paths.append(dst)
    return paths


def compare_faces(source_url, frame_paths):
    results = []
    for path in frame_paths:
        frame_url = storage.upload(path)
        raw = aigc.vision(FACE_COMPARE_PROMPT, media=[
            {"type": "image", "url": source_url},
            {"type": "image", "url": frame_url},
        ], max_tokens=256, json_mode=True)
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
    same = sum(bool(x.get("same_person")) for x in results)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "frames": results,
        "average_score": round(avg, 1),
        "same_person_frames": same,
        "total_frames": len(results),
        "passed": bool(results) and same / len(results) >= 0.75 and avg >= 70,
    }


def probe_video(path):
    err = run([ffmpeg(), "-hide_banner", "-i", path]).stderr
    duration = None
    for line in err.splitlines():
        if "Duration:" in line:
            try:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, sec = hms.split(":")
                duration = round(int(h) * 3600 + int(m) * 60 + float(sec), 3)
            except (ValueError, IndexError):
                pass
            break
    return {"duration_sec": duration, "probe": err[-1200:]}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


def run_round(round_no, source, source_url, desc):
    outdir = os.path.join(TRIAL_ROOT, "round_%03d" % round_no)
    os.makedirs(outdir, exist_ok=True)
    rec = {
        "round": round_no,
        "prompt_template": FACE_TEMPLATES[(round_no - 1) % len(FACE_TEMPLATES)],
        "prompt": line_prompt(desc, round_no),
        "video_prompt": VIDEO_PROMPT,
        "source": source,
        "person_description_slot": desc,
    }
    try:
        print("  生成线稿参考图...", flush=True)
        # 直接送出本轮模板，确保记录的 prompt 就是实际使用的 prompt。
        art_url = aigc.gen_image(rec["prompt"], size=None)
        art_path = storage.download(art_url, os.path.join(outdir, "line_art.jpg"))
        rec["line_art_url"] = art_url
        rec["line_art_file"] = art_path
        rec["line_art_judge"] = judge_line_art(art_url)
        if not rec["line_art_judge"].get("passed"):
            rec["status"] = "line_art_failed"
            save_json(os.path.join(outdir, "result.json"), rec)
            print("  线稿不通过：%s" %
                  rec["line_art_judge"].get("reason", ""), flush=True)
            return rec
    except Exception as exc:
        rec["status"] = "line_art_error"
        rec["error"] = str(exc)[:600]
        save_json(os.path.join(outdir, "result.json"), rec)
        print("  线稿失败：%s" % rec["error"], flush=True)
        return rec

    try:
        print("  Seedance 4 秒 480p...", flush=True)
        # 线稿图中的脸必须被声明为结构占位符；否则 Seedance 可能把白色
        # 线稿直接照搬到成片。gen_video_safe 还会在真人风控时自动回退。
        video_prompt = line_art.REAL_ACTOR_HINT + VIDEO_PROMPT
        video_result = line_art.gen_video_safe(
            video_prompt, ref_images=[art_url], duration_sec=4,
            ratio="9:16", resolution="480p")
        video_url = video_result["video_url"]
        video_path = storage.download(
            video_url, os.path.join(outdir, "seedance_480p.mp4"))
        rec["video_url"] = video_url
        rec["video_mode"] = video_result.get("mode")
        rec["video_file"] = video_path
        rec["video_probe"] = probe_video(video_path)
        frames = extract_frames(video_path, os.path.join(outdir, "frames"))
        rec["face_judge"] = compare_faces(source_url, frames)
        rec["status"] = "passed" if rec["face_judge"]["passed"] else "face_failed"
    except Exception as exc:
        rec["status"] = "video_error"
        rec["video_error"] = str(exc)[:600]
        rec["video_real_person_reject"] = line_art.is_real_person_reject(exc)
    save_json(os.path.join(outdir, "result.json"), rec)
    print("  本轮结果：%s" % rec["status"], flush=True)
    return rec


def run_fallback(source, source_url, desc):
    outdir = os.path.join(TRIAL_ROOT, "fallback_split")
    os.makedirs(outdir, exist_ok=True)
    rec = {
        "mode": "fallback_split_face_and_clothes",
        "face_prompt": FALLBACK_FACE_PROMPT,
        "clothes_prompt": FALLBACK_CLOTHES_PROMPT,
        "video_prompt": VIDEO_PROMPT,
        "person_description_slot": desc,
    }
    try:
        print("[fallback] 生成人脸线稿图...", flush=True)
        face_url = aigc.gen_image(FALLBACK_FACE_PROMPT, size=None)
        face_path = storage.download(face_url, os.path.join(outdir, "face_line_art.jpg"))
        print("[fallback] 生成独立服装参考图...", flush=True)
        clothes_url = aigc.gen_image(
            FALLBACK_CLOTHES_PROMPT + "人物外观参考：" + desc + "。")
        clothes_path = storage.download(
            clothes_url, os.path.join(outdir, "clothes_reference.jpg"))
        rec.update({
            "face_url": face_url, "face_file": face_path,
            "clothes_url": clothes_url, "clothes_file": clothes_path,
        })
        print("[fallback] 送入 Seedance 4 秒 480p...", flush=True)
        prompt = (
            "将 @图片1 定义为人物脸部结构线稿参考，将 @图片2 定义为同一人物的"
            "服装和体型参考；两张图共同定义 <主体1>。恢复 <主体1> 的自然真实人脸，"
            "保持 @图片2 中的发型、体型、服装和颜色连续一致。"
            + VIDEO_PROMPT)
        video_result = line_art.gen_video_safe(
            line_art.REAL_ACTOR_HINT + prompt,
            ref_images=[face_url, clothes_url], duration_sec=4,
            ratio="9:16", resolution="480p")
        video_url = video_result["video_url"]
        video_path = storage.download(
            video_url, os.path.join(outdir, "seedance_480p.mp4"))
        frames = extract_frames(video_path, os.path.join(outdir, "frames"))
        rec.update({
            "video_url": video_url, "video_mode": video_result.get("mode"),
            "video_file": video_path,
            "face_judge": compare_faces(source_url, frames),
            "status": "passed" if rec["face_judge"]["passed"] else "failed",
        })
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = str(exc)[:600]
    save_json(os.path.join(outdir, "result.json"), rec)
    return rec


def main():
    source = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not os.path.isfile(source):
        print("真人照片不存在：%s" % source)
        return 2
    os.makedirs(TRIAL_ROOT, exist_ok=True)
    source_url = storage.upload(source)
    print("输入：%s" % source, flush=True)
    print("提取人物外观描述...", flush=True)
    desc = describe_once(source)
    save_json(os.path.join(TRIAL_ROOT, "run_config.json"), {
        "source": source, "max_rounds": MAX_ROUNDS,
        "person_description_slot": desc,
        "face_templates": FACE_TEMPLATES,
        "video_prompt": VIDEO_PROMPT,
        "resolution": "480p", "duration_sec": 4,
    })

    summary = []
    for round_no in range(1, MAX_ROUNDS + 1):
        print("\n=== 第 %d/%d 轮 ===" % (round_no, MAX_ROUNDS), flush=True)
        round_dir = os.path.join(TRIAL_ROOT, "round_%03d" % round_no)
        rec = load_json(os.path.join(round_dir, "result.json"))
        if rec:
            print("  复用已保存结果：%s" % rec.get("status"), flush=True)
        else:
            rec = run_round(round_no, source, source_url, desc)
        summary.append({
            "round": round_no, "status": rec.get("status"),
            "line_art_judge": rec.get("line_art_judge"),
            "face_judge": rec.get("face_judge"),
            "dir": os.path.relpath(
                os.path.join(TRIAL_ROOT, "round_%03d" % round_no), config.OUTPUT_DIR),
        })
        save_json(os.path.join(TRIAL_ROOT, "summary.json"), {
            "source": source, "attempted": round_no,
            "max_rounds": MAX_ROUNDS, "results": summary,
        })
        if rec.get("status") == "passed":
            print("\n找到通过结果：第 %d 轮" % round_no, flush=True)
            return 0

    print("\n100 轮均未通过，进入拆分参考图备用方案。", flush=True)
    fallback = run_fallback(source, source_url, desc)
    save_json(os.path.join(TRIAL_ROOT, "summary.json"), {
        "source": source, "attempted": MAX_ROUNDS,
        "max_rounds": MAX_ROUNDS, "results": summary,
        "fallback": fallback,
    })
    return 0 if fallback.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
