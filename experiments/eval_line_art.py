"""线稿 prompt 迭代评测：每轮跑 10 张，用 VLM 判「只有脸是线稿、其余仍写实」。

跑法：python3 eval_line_art.py [round_tag]
- 10 张输入真人照只生成一次，缓存在 output/line_art_eval/inputs/
- 每轮把当前 line_art.FACE_LINE_ART 的效果跑一遍，结果存 output/line_art_eval/{round_tag}/
- result.json 里落 prompt 快照 + 每张的判定 + 通过率，10/10 才算达标
"""

# 从仓库根目录跑：python experiments/eval_line_art.py（下面两行把仓库根加进 sys.path，这样 import aigc / line_art 才找得到）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import concurrent.futures as cf
import json
import os
import sys

import aigc
import config
import line_art
import storage

ROOT = os.path.join(config.OUTPUT_DIR, "line_art_eval")
INPUTS = os.path.join(ROOT, "inputs")

# 10 个多样化的真人形象，覆盖性别/年龄/景别/背景/服装，避免只在单一样本上过拟合
PERSONS = [
    "一位亚洲年轻女性，黑色长直发，白色针织衫，正面半身，浅灰纯色背景",
    "一位亚洲中年男性，短发，深蓝衬衫，正面半身，办公室虚化背景",
    "一位亚洲年轻男性，寸头，黑色卫衣，正面胸部以上特写，白色背景",
    "一位亚洲女性，棕色卷发扎起，米色风衣，全身站姿，街道虚化背景",
    "一位亚洲年长女性，短卷白发，紫色开衫，正面半身，居家客厅背景",
    "一位亚洲女性，双手捧着一只白色马克杯，浅蓝毛衣，正面半身，木质桌面前",
    "一位亚洲男性，戴黑框眼镜，灰色西装外套，四分之三侧脸半身，深灰背景",
    "一位亚洲女性运动装扮，马尾辫，黑色运动背心，正面半身，健身房虚化背景",
    "一位亚洲女性，手里拿着一支口红对着镜头展示，长发披肩，正面半身，浅粉背景",
    "一位亚洲男性厨师，白色厨师服，戴帽，正面半身，厨房虚化背景",
]

PERSON_TMPL = ("真人实拍摄影：%s。真实相机质感——真实皮肤纹理、真实发丝、真实布料材质与自然光影，"
               "写实照片而非插画。均匀柔光，画面只有一个人物，无文字无水印。")

JUDGE_SYSTEM = "你是严格的图像质检员，只按看到的画面回答，输出 json。"
JUDGE_USER = (
    "检查这张人物图，判断「线稿改造」是否只作用在面部、且形状贴合脸型。逐项看清楚再回答：\n"
    "1. face_is_line_art：面部（眉眼鼻嘴、脸型轮廓）是否为黑色线条描绘、皮肤留白的线稿？\n"
    "2. hair_is_photoreal：头发是否仍是写实照片质感（有真实发丝光泽和体积），而不是线条勾勒？\n"
    "3. clothes_is_photoreal：衣服是否仍是写实照片质感（有真实布料材质、颜色和褶皱阴影），"
    "而不是线条勾勒的白色轮廓？\n"
    "4. skin_is_photoreal：颈部/双手等裸露皮肤是否仍是写实真实肤色，而不是留白线稿？\n"
    "5. background_is_photoreal：背景是否仍是原有的写实背景，而不是变成白纸？\n"
    "6. mask_fits_face：白色留白区域的边界是否严格沿着脸型轮廓走（上到发际线、"
    "两侧到脸颊边缘、下到下颌线，整体是脸的形状）？如果白色区域是方形/矩形/圆角矩形，"
    "或有平直的切边，或盖住了头发、额头上方的头发、耳朵、颈部、背景，则为 false。\n"
    "只要衣服/头发/皮肤/背景中任何一处被画成了线稿或变成留白线条图，对应项就是 false。\n"
    '严格输出 json：{"face_is_line_art":bool,"hair_is_photoreal":bool,'
    '"clothes_is_photoreal":bool,"skin_is_photoreal":bool,"background_is_photoreal":bool,'
    '"mask_fits_face":bool,"reason":"一句话说明哪里不对"}'
)

CHECKS = ("face_is_line_art", "hair_is_photoreal", "clothes_is_photoreal",
          "skin_is_photoreal", "background_is_photoreal", "mask_fits_face")


def ensure_inputs():
    """10 张输入真人照，已存在就复用（省钱省时，也保证跨轮次可比）。"""
    os.makedirs(INPUTS, exist_ok=True)
    paths = [os.path.join(INPUTS, "p%02d.jpg" % (i + 1)) for i in range(len(PERSONS))]
    todo = [(i, p) for i, p in enumerate(paths) if not os.path.isfile(p)]
    if todo:
        print("生成 %d 张输入真人照..." % len(todo))

        def gen(item):
            i, path = item
            storage.download(aigc.gen_image(PERSON_TMPL % PERSONS[i]), path)
            print("  input", os.path.basename(path))

        with cf.ThreadPoolExecutor(5) as ex:
            list(ex.map(gen, todo))
    return paths


def judge(url: str) -> dict:
    raw = aigc.vision(JUDGE_USER, media=[{"type": "image", "url": url}],
                      system=JUDGE_SYSTEM, json_mode=True)
    txt = raw.strip().strip("`")
    if txt.startswith("json"):
        txt = txt[4:]
    obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
    obj["passed"] = all(bool(obj.get(k)) for k in CHECKS)
    return obj


def run_one(args):
    idx, src, outdir, quiet = args
    name = "p%02d" % (idx + 1)
    try:
        url = line_art.to_line_art(src)
        storage.download(url, os.path.join(outdir, name + "_lineart.jpg"))
        v = judge(url)
        if not quiet:
            print("  %s %s %s" % (name, "PASS" if v["passed"] else "FAIL",
                                  "" if v["passed"] else v.get("reason", "")[:60]))
        return {"name": name, "input": os.path.basename(src), "url": url, **v}
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print("  %s ERROR %s" % (name, str(exc)[:120]))
        return {"name": name, "input": os.path.basename(src), "error": str(exc)[:400],
                "passed": False}


def evaluate(tag: str, quiet: bool = False) -> dict:
    """按当前 line_art 里的 prompt 跑一轮 10 张，返回评测记录。"""
    outdir = os.path.join(ROOT, tag)
    os.makedirs(outdir, exist_ok=True)
    srcs = ensure_inputs()
    if not quiet:
        print("[%s] 跑 %d 张..." % (tag, len(srcs)))
    with cf.ThreadPoolExecutor(5) as ex:
        results = list(ex.map(run_one, [(i, s, outdir, quiet) for i, s in enumerate(srcs)]))

    passed = sum(1 for r in results if r.get("passed"))
    rec = {"round": tag, "passed": passed, "total": len(results),
           "prompt": {"redraw_base": line_art._REDRAW_BASE,
                      "face_line_art": line_art.FACE_LINE_ART},
           "fail_counts": {k: sum(1 for r in results if not r.get(k, True)) for k in CHECKS},
           "results": results}
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    return rec


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "round1"
    rec = evaluate(tag)
    print("\n[%s] %d/%d 通过" % (tag, rec["passed"], rec["total"]))
    print("各项失败数:", rec["fail_counts"])
    print("产物:", os.path.join(ROOT, tag))
    return 0 if rec["passed"] == rec["total"] else 1



if __name__ == "__main__":
    sys.exit(main())
