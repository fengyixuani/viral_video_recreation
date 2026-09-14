"""商品事实卡（步骤 product）：把用户填的信息 + 商品图读成一张事实卡。

事实卡是下游写剧本、生图生视频的唯一商品口径（name / 卖点 / 外观 / images / image_urls）。
商品图链路在 product_images.py，这里只做「事实归集」。
"""
import json
import os

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from product_images import (harvest_product_images, _index_product_images,
                            _refine_user_images, _select_refs)
from task_store import _p, _rel, log


FACT_PROMPT = """整理一张商品事实卡，输出 json。

【用户已提供的信息】（这些是事实，必须原样保留，不要改写）
__GIVEN__

__IMAGE_HINT__
输出 json：
{"name":"商品名","category":"品类",
 "selling_points":["3-4条卖点，具体可感知"],
 "copy":"一句话广告语，10-15字",
 "audience":"目标人群",
 "appearance":"商品外观：外形结构、材质质感、主色配色、包装显著文字或图案，60字内",
 "usage_scene":"典型使用场景",
 "inferred_fields":["哪些字段是你推断出来的，而不是用户给的"]}
用户给了卖点就以用户的为准，最多补到 4 条；不要编造检测数据、专利、功效或品牌背书。
「appearance」只写图片里实际看得见的：屏幕/表盘当前显示什么就写什么；
图上营销文案宣称的功能画面（壁纸角色、界面动效）不属于外观，不要写进 appearance。
这类文字功能可以概括成卖点，但要写成「支持××」的功能宣称，不要描述成画面里存在的形象。
只输出 json。"""


def _material_clues(rec: dict, harvest: dict) -> str:
    """没有商品图时给 LLM 的素材线索：抽帧判定 + 素材口播里的原话。"""
    bits = []
    judge = harvest.get("判定") or {}
    for k in ("商品", "外观", "卖点"):
        if judge.get(k):
            bits.append("素材里看到的%s：%s" % (k, json.dumps(judge[k], ensure_ascii=False)
                                               if isinstance(judge[k], list) else judge[k]))
    idx_path = _p(rec["task_id"], "assets", "material_index.json")
    if os.path.isfile(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            segs = (json.load(fh) or {}).get("片段") or []
        lines = [str(s.get("声音台词") or "").strip() for s in segs]
        lines = [t for t in lines if len(t) > 4][:20]
        if lines:
            bits.append("素材口播原话（商品名与卖点尽量从这里提取，不要编造）：\n" + "\n".join(lines))
    return ("\n".join(bits) + "\n") if bits else ""


def step_product(rec: dict) -> dict:
    given = {k: v for k, v in rec["product"].items() if v}
    images = list(rec["inputs"].get("product_images") or [])
    img_check, img_index, img_sel = [], [], {}
    if images:
        # 商家图片素材先逐张结构化理解 → 按理解选参考图并排序 → 再判断选中的要不要编辑。
        # 顺序很重要：@图片1 必须是完整全貌，主体小/牛皮癣/大字报的图直接当参考图，
        # 生成端会照抄这些干扰。
        img_index = _index_product_images(rec, images)
        images, img_sel = _select_refs(rec, img_index)
        roles = {c.get("文件"): "%s（%s）" % (c.get("角色") or "", c.get("理由") or "")
                 for c in (img_sel.get("选中") or []) if c.get("文件")}
        images, img_check = _refine_user_images(
            rec, images, index=img_index, limit=len(img_sel.get("选中") or []) or None,
            roles=roles)
    harvest = {} if images else harvest_product_images(rec)
    if harvest:
        # 白底图排在前面：它是干净的商品主图，抽帧留着当兜底与人工核对用。
        # 兜底那张必须取「最佳帧」而不是「抽帧」的第一张：抽帧顺序是挑片段模型给的候选
        # 顺序，跟质量无关（实测 c1b4：第一张是耳塞被两指捏成尖角的帧，判定已经说了
        # 不清晰、有手，照样被当成商品参考图进了事实卡，生成端照着它画出变形的商品）。
        best = harvest.get("最佳帧")
        images = (harvest.get("白底图") or []) + ([best] if best else [])
    if images and harvest:
        hint = ("这些图是从用户素材里抽帧、并据此生成的白底商品图，商品外观以图为准。\n"
                + _material_clues(rec, harvest))
    elif images:
        hint = "这是商品图，结合图片补全外观与使用场景。\n"
    else:
        hint = ("没有商品图，只能依据上面的信息推断外观，不确定就写「不确定」。\n"
                + _material_clues(rec, harvest))
    raw = aigc.understand(
        FACT_PROMPT.replace("__GIVEN__", json.dumps(given, ensure_ascii=False, indent=1))
                   .replace("__IMAGE_HINT__", hint),
        media=[{"type": "image", "url": p} for p in images[:3]] or None,
        max_tokens=2048, json_mode=True)
    info = write_script._parse_json(raw)
    info["name"] = rec["product"].get("name") or info.get("name") or "未命名商品"
    if rec["product"].get("selling_points"):
        info["selling_points"] = rec["product"]["selling_points"]
    if rec["product"].get("category"):
        info["category"] = rec["product"]["category"]
    info["provided_fields"] = sorted(given)
    info["images"] = images
    # 参考图只带前 PRODUCT_REF_MAX 张：提示词里的 @图片N 必须与实际下发的参考图一一对应，
    # 多出来的图仍留在 images 里（事实卡可查），不静默丢。
    refs = images[:rules.PRODUCT_REF_MAX]
    info["image_urls"] = [storage.upload(p) for p in refs]
    if len(images) > len(refs):
        info["参考图截断"] = ("商品图 %d 张，只有前 %d 张当生成参考图（模型参考图上限 %d）"
                             % (len(images), len(refs), rules.PRODUCT_REF_MAX))
    if img_check:
        fixed = sum(1 for d in img_check if d.get("最终") != d.get("原图"))
        info["商品图来源"] = ("用户上传（已体检，%d/%d 张改造后使用）"
                             % (fixed, len(img_check)) if fixed else "用户上传（已体检）")
        info["商品图检查"] = img_check
    if img_index:
        info["商品图索引"] = img_index
        # 编号 → 最终文件 → 公网 URL 的对照表：分镜级选参考图要按编号取图，
        # 而 images 里可能已经是改造后的文件（编号只在索引里），两头必须能对上。
        origin = {c.get("最终"): c.get("原图") for c in img_check if c.get("最终")}
        vetted = {c.get("最终") for c in img_check if c.get("最终")}
        no_of = {d["文件"]: d.get("编号") for d in img_index}
        url_of = dict(zip(refs, info["image_urls"]))
        # 已体检=False 的是没进全局参考图、也就没做编辑判断的图：分镜级选图选中它时会补体检
        info["商品图明细"] = [{"编号": no_of.get(origin.get(p, p)), "文件": p,
                              "原图": origin.get(p, p), "url": url_of.get(p),
                              "已体检": p in vetted}
                             for p in images]
    if img_sel:
        info["商品图选图"] = img_sel
    if harvest:
        # harvest_product_images 即使一帧都没抽出来也返回 {"抽帧": [], "白底图": [], ...}，
        # 非空字典恒为真——不看实际产出就标「抽帧 + 白底图生成」，报告和前端会显示一个
        # 根本不存在的来源。所以按真实拿到的图报，一张都没有就直接说清楚。
        got = ["白底图 %d 张" % len(harvest.get("白底图") or [])] if harvest.get("白底图") else []
        if harvest.get("抽帧"):
            got.append("素材抽帧 %d 张" % len(harvest["抽帧"]))
        info["商品图来源"] = ("用户素材抽帧 + 白底图生成（%s）" % "、".join(got) if got
                             else "用户素材抽帧未取到任何可用图（事实卡无商品图）")
        info["商品图补齐"] = {k: v for k, v in harvest.items() if k != "最佳帧"}
    path = _p(rec["task_id"], "product", "fact_card.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    log(rec, "商品事实卡：%s（卖点 %d 条，商品图 %d 张%s）"
        % (info["name"], len(info.get("selling_points") or []), len(images),
           "，来自素材补齐" if harvest else ""))
    if info.get("参考图截断"):
        log(rec, "  %s" % info["参考图截断"])
    return {"artifact": _rel(rec["task_id"], path), "product": info["name"],
            "images": len(images), "harvested": bool(harvest)}
