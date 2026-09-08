"""分段出片（步骤 generate）：段内按镜头混合，能裁的裁素材，裁不动的交 seedance 补。

- 直接裁剪：_cut_piece 从用户素材切片裁画面（窗口游标 _ClipWindows 防同片段多镜重复取用）
- 素材编辑 / 参考图生视频：_gen_piece，参考图按分镜计划下发，真人风控走 line_art 降级
- 补片走不通且素材本来能出镜时退回裁素材：用户素材优先于 AI 重演
"""
import concurrent.futures as cf
import json
import os
import shutil
import threading

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import config  # pyright: ignore[reportImplicitRelativeImport]
import line_art  # pyright: ignore[reportImplicitRelativeImport]
import produce_video  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from media import (MAX_SLOWDOWN, _ffmpeg, _sec, cut_clip, extract_frame)
from product_images import _make_state_images, _piece_product_urls, _plan_shot_refs
from ref_audio import separate_bgm
from shot_match import CLIP_ACTIONS, _can_cut, _cuttable, _shot_text
from task_store import _d, _p, _rel, log
from voice_dub import (DUB_MAX_TEMPO, _dub_scale, _dub_wav, _fill_silent_dub, _join_lines,
                       _paste_dub, _voice_from_piece, _voice_plan)

# 视频编辑用的参考片段下限：seedance r2v 硬要求 ≥1.8s，但太短的片段控不住整段画面，
# 要 >5s 才有参考价值。凑不到 5s 就不做编辑，改用参考图生视频靠提示词还原画面。
REF_CLIP_MIN = 5.0


# ---------------- 步骤 7：分段出片 ----------------
EDIT_HINT = ("以 @视频1 为基础进行编辑：严格保留 @视频1 的镜头运动、构图与画面节奏，"
             "把其中的商品替换为参考商品图所示商品，人物呈现真实自然的五官。")


def _video_rejected(exc) -> bool:
    """风控拒的是**输入视频**（"input video 'content[1]' may contain real person"）。

    line_art.gen_video_safe 的三级降级只换 ref_images，参考视频一直留在请求里，
    所以这种拒绝三次全废、整块必失败。判出来后由 _gen_piece 丢掉参考视频重生成。
    """
    return "input video" in str(exc).lower() and line_art.is_real_person_reject(exc)


def _audio_rejected(exc) -> bool:
    """拒的是**参考音色**：格式不合（"audio format ... is not valid ... in r2v"），
    或者参考音成了唯一的参考输入（"reference_audio cannot be the only reference input"）。

    seedance 的 reference_audio 只收 wav/mp3、单段 2~15s，而且必须搭配参考图/参考视频；
    不合格时整条请求 41000000 失败。判出来后丢掉参考音色重生成，只让音色退化，别丢整块画面。
    """
    msg = str(exc).lower()
    return "audio format" in msg or "reference_audio cannot be the only" in msg


def _seg_matches(seg: dict, matches: list) -> list:
    by_idx = {str(m["序号"]): m for m in matches}
    return [by_idx.get(str(i)) for i in (seg.get("镜头序号") or [])]


def _clip_audio(clip: str, action: str) -> dict:
    """按「用户切片处理」表的动作处理已裁好的切片音轨（原地替换 clip 的音轨）。

    静音走 anullsrc 而不是 -an：整段是 concat 拼起来的，少一条音轨会直接拼不上。
    """
    todo = CLIP_ACTIONS.get(action, "keep")
    if todo in (None, "keep"):
        return {"音轨": "原声保留" if todo == "keep" else "未处理"}
    stem = os.path.splitext(clip)[0]
    tmp, info = stem + "_a.mp4", {}
    if todo == "mute":
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", clip,
               "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
        cmd += produce_video.aac_args() + ["-shortest", tmp]
        info["音轨"] = "静音"
    else:
        out = separate_bgm(clip, stem + "_vocal.m4a", want="vocal")
        info = {"音轨": "分离BGM保留人声", "分离方法": out["method"]}
        if out.get("fallback_reason"):
            info["分离回落原因"] = out["fallback_reason"]
        cmd = [_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", clip,
               "-i", out["file"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
        cmd += produce_video.aac_args() + ["-shortest", tmp]
    ret = produce_video._run(cmd)
    if ret.returncode != 0 or not os.path.isfile(tmp):
        raise RuntimeError("切片音轨处理失败（%s）：%s" % (action, ret.stderr[-200:]))
    os.replace(tmp, clip)
    return info


# 这些切片动作的画面能直接出镜（「不使用」只能当素材编辑的参考视频，见 step_match 的降级）
def _demote_edit_rows(rec: dict, matches: list, pool: dict) -> int:
    """「素材编辑」的前提不成立时降级成「直接裁剪」，而不是掉到纯生成。返回降级条数。

    素材编辑要拿这条素材当参考视频，前提是它**没有真人出镜**（有真人必被 seedance 风控拒）
    且长度够 REF_CLIP_MIN。前提不成立时旧实现直接走参考图生视频——等于把这条已经通过
    `画面匹配` 门槛的真素材整条丢掉换成 AI 重演（实测 5 个镜头里 5 个都这么丢的）。
    匹配度 0.55~0.8 只说明「改造一下更好」，不代表画面不能用，所以宁可原样裁进成片。
    """
    n = 0
    for row in matches:
        if row.get("策略") != "素材编辑" or not _cuttable(row):
            continue
        meta = pool.get(row.get("片段ID")) or {}
        if (not _has_person(meta)
                and float(row.get("可用秒") or 0) * MAX_SLOWDOWN >= REF_CLIP_MIN):
            continue                      # 前提成立，留给素材编辑
        row["策略"] = "直接裁剪"
        row["策略降级"] = "素材编辑→直接裁剪：这条素材当不了参考视频（真人出镜或太短），画面已过门槛，直接裁剪出镜"
        n += 1
    if n:
        log(rec, "素材编辑降级为直接裁剪 %d 镜（当不了参考视频，但画面能直接用）" % n)
    return n


def _seg_blocks(seg: dict, rows: list) -> list:
    """把一段按「能不能裁用户素材」切成连续的块：[{"kind": "cut"|"gen", "rows", "镜头"}]。

    用户素材为主、AI 补片为辅：能裁的镜头一律裁，裁不动的**连续**镜头并成一块交给一次
    生成。必须并块——seedance 最短出 4s，一镜一块会把 0.4s 的镜头也撑成 4s，
    既费额度又打乱参考片的节奏。

    裁剪块还要按「需要配音」再分一层：配音是整块贴的（_dub_wav 念的是整块台词，
    _paste_dub 用 -map 0:v:0 -map [a] 换掉整块音轨），块内混着「原声直接使用 / 分离BGM
    保留人声」的镜头时，那一镜的人声与画面内音效会被一起丢掉，TTS 还按整块时长铺开、
    口播与画面对不上。裁剪本来就是逐镜裁完再拼（见 _cut_piece），多分一块不多花额度。
    生成块不能这么分——那正好破坏上面的并块前提。
    """
    blocks = []
    for shot_no, row in zip(seg.get("镜头序号") or [], rows):
        kind = "cut" if _can_cut(row) else "gen"
        dub = bool(row.get("需要配音")) if kind == "cut" else None
        if blocks and blocks[-1]["kind"] == kind and blocks[-1]["需要配音"] == dub:
            blocks[-1]["rows"].append(row)
            blocks[-1]["镜头"].append(shot_no)
        else:
            blocks.append({"kind": kind, "rows": [row], "镜头": [shot_no], "需要配音": dub})
    return blocks


def _concat_same(parts: list, out: str, listfile: str) -> None:
    """顺序拼接同格式片段（cut_clip 的产物已统一分辨率/帧率/编码，可以直接复制流）。"""
    if len(parts) == 1:
        shutil.copyfile(parts[0], out)
        return
    with open(listfile, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", out])
    if ret.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError("素材片段拼接失败：%s" % ret.stderr[-300:])


def _block_planned(block: dict, shots: dict) -> float:
    """这一块按剧本该占多长（各镜头计划时长之和）。"""
    return sum(max(0.4, float((shots.get(n) or {}).get("时长秒") or 2)) for n in block["镜头"])


def _block_needs(block: dict, shots: dict, scale: float = 1.0) -> list:
    """这一块每镜要裁多少秒：按计划时长（×放长倍数）分配，缺口挪给还有余料的镜。

    某镜的素材不够长时 cut_clip 只能放慢到 MAX_SLOWDOWN 倍，还是补不齐它那一份；
    这时整块就比计划短，念白也就塞不进去（实测有一镜只有 4s 素材、计划要 8.4s，
    整块短 2s，念白被砍掉 2s）。块内其他镜往往还有没用到的素材，缺口挪给它们即可——
    块的总长对齐计划才是目的，镜与镜之间的比例可以让一让。
    """
    rows = block["rows"]
    need = [max(0.4, float((shots.get(n) or {}).get("时长秒") or r.get("时长秒") or 2))
            * max(1.0, float(scale or 1.0)) for n, r in zip(block["镜头"], rows)]
    # 放慢到顶时这一镜最多能撑多长
    cap = [float(r.get("可用秒") or 0) * MAX_SLOWDOWN for r in rows]
    gap = sum(max(0.0, x - c) for x, c in zip(need, cap))
    room = [max(0.0, c - x) for x, c in zip(need, cap)]
    if gap > 0.05 and sum(room) > 0.05:
        take = min(gap, sum(room))
        need = [x + r * take / sum(room) for x, r in zip(need, room)]
    return need


class _ClipWindows:
    """同一素材片段被多个镜头命中时，错开各镜头的取用窗口。

    不加它的话：多个分镜命中同一片段时每刀都从片段窗口起点裁（段1 取 0-6.2s、
    段2 取 0-3.2s），成片同画面出现两次。这里按片段 ID 维护游标：每裁一刀游标
    前移，下一刀从没用过的位置接着取；新鲜画面不足 1s 才回卷到起点并显式标记。
    同一镜头因配音放长而重裁（key 相同）复用原起点，只向后延长，不再另开窗口。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cursor = {}   # 片段ID -> 窗口内已用秒
        self._grant = {}    # "tag#镜序" -> 窗口内起点（重裁复用）

    def take(self, key: str, row: dict, need: float):
        """返回 (窗口内偏移, 可取秒数, 是否回卷复用)。"""
        pid = row.get("片段ID") or row.get("源文件") or ""
        avail = max(0.4, _sec(row.get("可用秒")))
        with self._lock:
            off, wrapped = self._grant.get(key), False
            if off is None:
                cur = self._cursor.get(pid, 0.0)
                if avail - cur >= min(need, 1.0):
                    off = cur
                else:
                    off, wrapped = 0.0, cur > 0
                self._grant[key] = off
            take = max(0.4, min(need, avail - off))
            self._cursor[pid] = max(self._cursor.get(pid, 0.0), off + take)
            return off, take, wrapped


def _cut_piece(rec: dict, block: dict, shots: dict, segdir: str, tag: str,
               scale: float = 1.0, alloc: "_ClipWindows" = None) -> dict:
    """一块连续的「直接裁剪」镜头：逐镜裁剪用户素材后拼成一个片。

    scale > 1 表示这一块要放长（配音塞不进计划时长时用，见 _dub_scale）：优先多取素材里
    还没用到的那几秒真画面，素材本身不够长时 cut_clip 才放慢补足。
    """
    tid = rec["task_id"]
    parts, detail = [], []
    needs = _block_needs(block, shots, scale)
    for i, (shot_no, row, need) in enumerate(zip(block["镜头"], block["rows"], needs), 1):
        dst = _p(tid, "generated", "clips", "seg%s_%02d.mp4" % (tag, i))
        base = _sec(row.get("开始秒"))
        off, take, wrapped = (alloc.take("%s#%d" % (tag, i), row, need) if alloc
                              else (0.0, _sec(row.get("可用秒")), False))
        if wrapped:
            log(rec, "  素材片段 %s 新鲜画面用尽，镜%s 回卷复用起点（成片会出现重复画面）"
                % (row.get("片段ID"), shot_no))
        cut = cut_clip(row["源文件"], base + off, take, need, dst)
        cut.update(_clip_audio(dst, row.get("切片动作") or "原声直接使用"))
        cut.update({"镜头序号": shot_no, "片段ID": row["片段ID"], "匹配度": row["匹配度"],
                    "切片动作": row.get("切片动作"), "需要配音": row.get("需要配音"),
                    "源窗口": [round(base + off, 2), round(base + off + take, 2)]})
        if wrapped:
            cut["复用回卷"] = True
        parts.append(dst)
        detail.append(cut)
    out = os.path.join(segdir, "seg_%s.mp4" % tag)
    _concat_same(parts, out, _p(tid, "generated", "clips", "seg%s_list.txt" % tag))
    piece = {"kind": "cut", "mode": "cut_user_video", "file": out, "镜头": list(block["镜头"]),
             "片段": detail, "需要配音": any(d.get("需要配音") for d in detail),
             "时长秒": round(produce_video._duration(out), 2)}
    if scale > 1.005:
        piece["放长倍数"] = round(scale, 3)
    return piece


# 参考视频里出现真人就会被 seedance 真人风控拒（40000002 input video may contain real person），
# 而且 gen_video_safe 的降级只换参考图、不会丢视频，一拒就整段失败。所以真人出镜的素材
# 不做视频编辑，只走参考图生视频靠提示词还原；手部/产品特写这类没有人物的才可以当参考视频。
PERSON_WORDS = ("人物", "真人", "出镜", "人脸", "面部", "脸", "女性", "男性", "女生", "男生",
                "女子", "男子", "女孩", "男孩", "少女", "博主", "主播", "模特", "演员")


def _has_person(meta: dict) -> bool:
    """素材片段里是否有真人出镜。

    事实来自 analyze_materials.JUDGE_FIELDS 的「真人出镜」布尔；标注缺失（None）时
    才回落到关键词，并按最坏情况处理——宁可不做编辑，也不要撞真人风控让整段失败。
    """
    if not meta:
        return True          # 没有标注信息时按最坏情况处理，宁可不做编辑
    flag = rules.as_bool(meta.get("真人出镜"))
    if flag is not None:
        return flag
    parts = [meta.get("主体"), meta.get("画面"), meta.get("视觉标签"), meta.get("内容标签")]
    text = " ".join(x if isinstance(x, str) else " ".join(x or []) for x in parts)
    return any(w in text for w in PERSON_WORDS)


def _pick_ref_clip(rec: dict, rows: list, shots: dict, pool: dict, tag: str, label: str):
    """给「素材编辑」挑一条能用的参考片段与其来源信息，挑不到返回 (None, {})（改走参考图生视频）。

    两道门槛：素材里不能有真人出镜（否则必被风控拒），长度不能短于 REF_CLIP_MIN
    （太短控不住整段画面，还会撞 r2v 的 1.8s 硬限制）。
    """
    tid = rec["task_id"]
    cands = sorted((r for r in rows if r and r.get("匹配度", 0) >= rules.EDIT_SCORE and r.get("源文件")),
                   key=lambda r: -r["匹配度"])
    for r in cands:
        if _has_person(pool.get(r.get("片段ID"))):
            log(rec, "  %s 镜%s 的素材有真人出镜，不做视频编辑" % (label, r["序号"]))
            continue
        avail = float(r.get("可用秒") or 0)
        if avail * MAX_SLOWDOWN < REF_CLIP_MIN:      # 放慢到极限也凑不够长度
            log(rec, "  %s 镜%s 的素材只有 %.1fs，凑不到 %.0fs"
                % (label, r["序号"], avail, REF_CLIP_MIN))
            continue
        need = max(REF_CLIP_MIN, float(shots.get(r["序号"], {}).get("时长秒") or 0))
        clip = _p(tid, "generated", "clips", "ref%s.mp4" % tag)
        info = cut_clip(r["源文件"], r["开始秒"], avail, need, clip)
        if info["duration_sec"] + 0.05 < REF_CLIP_MIN:
            continue
        log(rec, "  %s 素材编辑：参考片段取自镜%s（%.1fs，匹配度 %.2f，无真人出镜）"
            % (label, r["序号"], info["duration_sec"], r["匹配度"]))
        # 来源一并返回并记进 segments.json：报告里要能回答「@视频1 是哪条素材的哪几秒」
        return clip, {"片段ID": r.get("片段ID"), "镜头": r["序号"], "源文件": r["源文件"],
                      "源窗口": [round(r["开始秒"], 2),
                                 round(r["开始秒"] + info["take_sec"], 2)],
                      "匹配度": r["匹配度"], "时长秒": info["duration_sec"]}
    if cands:
        log(rec, "  %s 没有可用作参考视频的素材，改用参考图生视频靠提示词还原" % label)
    return None, {}


# 动作名（rules.GAP_RULES 的「动作」）→ 这一段的模特怎么来；实现见 _gap_refs。
GAP_ACTIONS = {"AIGC直接生成": "product_only", "提取模特改线稿图": "extract_line_art",
               "直接生成线稿图": "shared_line_art", "文字描述生成": "text_only"}


def _model_frame_line_art(rec: dict, matches: list, pool: dict) -> str:
    """从已选中的用户切片里抽一帧有模特人脸的画面，转成线稿图 URL（整任务只做一次）。

    线稿是必须的：seedance 拒收含真人人脸的参考图（见 line_art 模块说明），
    而线稿脸能把人物锚到真实模特身上又能过风控。
    """
    tid = rec["task_id"]
    cache = _p(tid, "generated", "model_ref.json")
    if os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh).get("url") or ""
    url, note = "", ""
    for row in sorted((r for r in matches if r.get("源文件") and r.get("片段ID")),
                      key=lambda r: -r.get("匹配度", 0)):
        meta = pool.get(row["片段ID"]) or {}
        if rules.as_bool(meta.get("模特人脸出镜")) is not True:
            continue
        try:
            frame = extract_frame(row["源文件"], row["开始秒"] + 0.2,
                                  _p(tid, "generated", "model_frame.jpg"))
            url = line_art.to_line_art(frame)
            note = "取自 %s" % row["片段ID"]
        except (RuntimeError, OSError) as exc:
            note = "抽帧/线稿失败：%s" % str(exc)[:150]
        break
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"url": url, "说明": note}, fh, ensure_ascii=False, indent=2)
    log(rec, "模特线稿参考图：%s" % (note or "已选切片里没有模特人脸"))
    return url


def _refs_with(head: list, product_urls: list) -> tuple:
    """参考图 = 人物/线稿图在前 + 商品图在后，总数不超过 PRODUCT_REF_MAX。

    同时返回实际带上的商品图，供 _asset_manifest 编号——清单里的 @图片N 必须与真正下发的
    参考图一一对应，否则提示词会引用一张没传的图。
    """
    prods = list(product_urls)[:max(0, rules.PRODUCT_REF_MAX - len(head))]
    return (list(head) + prods)[:rules.PRODUCT_REF_MAX], prods


def _gap_refs(rec: dict, seg: dict, built: dict, product_urls: list, gap: dict,
              model_url: str) -> dict:
    """按「切片缺失补片」表的动作决定参考图与要不要补人物文字描述。

    返回 {"refs","extra","char_ids","线稿人物","product_urls"}；线稿人物=True 时提示词要前置
    line_art.REAL_ACTOR_HINT，否则线稿脸会被照抄进成片。
    """
    cmap = {c.get("编号"): c for c in (built.get("素材") or {}).get("人物") or []
            if c.get("url")}
    char_ids = [c for c in (seg.get("人物编号") or []) if c in cmap]
    todo = GAP_ACTIONS.get(gap["动作"], "product_only")
    if todo == "extract_line_art" and model_url:
        refs, prods = _refs_with([model_url], product_urls)
        return {"refs": refs, "extra": "", "char_ids": char_ids, "线稿人物": True,
                "product_urls": prods}
    if todo == "shared_line_art" and char_ids:
        refs, prods = _refs_with([cmap[c]["url"] for c in char_ids], product_urls)
        return {"refs": refs, "extra": "", "char_ids": char_ids, "线稿人物": True,
                "product_urls": prods}
    if todo == "text_only":
        looks = [c.get("外观") or "" for c in built["剧本"].get("人物设定") or []
                 if c.get("编号") in (seg.get("人物编号") or [])]
        extra = ("画面中的人物外观：%s。" % "；".join(x for x in looks if x)) if looks else ""
        refs, prods = _refs_with([], product_urls)
        return {"refs": refs, "extra": extra, "char_ids": [], "线稿人物": False,
                "product_urls": prods}
    refs, prods = _refs_with([], product_urls)
    return {"refs": refs, "extra": "", "char_ids": [], "线稿人物": False,
            "product_urls": prods}


def _sub_seg(seg: dict, block: dict, shots: dict) -> dict:
    """把段内要补片的若干镜头包成一个「虚拟段」交给生成。

    视频提示词换成这些镜头自己的画面：段级提示词写的是整段（含已经用用户素材裁掉的镜头），
    照抄会让补片把裁掉的画面再演一遍。生成时长按 seedance 下限 4s 兜，超出计划的部分
    生成后裁回（见 _trim_to），否则短块会把整段节奏撑长。
    """
    group = [shots.get(i) or {} for i in block["镜头"]]
    need = sum(max(0.1, float(s.get("时长秒") or 0)) for s in group)
    sub = dict(seg)
    sub.update({"镜头序号": list(block["镜头"]),
                "台词": [t for t in (_shot_text(s) for s in group) if t],
                "原始时长秒": round(need, 1), "生成时长秒": max(4, int(round(need))),
                "视频提示词": "；".join(x for x in (s.get("画面") for s in group) if x)
                              or seg.get("视频提示词") or ""})
    return sub


PIECE_CHECK_PROMPT = """这是刚生成的一小段广告视频，验收它能不能用。

【这一块的分镜要求】
__SHOTS__

【商品】__NAME__：__LOOK__
__STATES__
只按下面四条判不合格，其余（光影、构图、演绎自由度、与要求略有出入）都放行：
1. 商品明显不对：变成别的东西、结构或配色严重走样、关键部件凭空增减。判之前先看分镜要求：
   分镜本来就把商品演成不完整形态（装配、拆解、部件单独出现、只拍局部）时，缺部件、缺印刷
   文字是对的，不算走样；已经出现的那部分仍要与商品外观一致
2. 画面与分镜要求完全无关
3. 商品上原本清晰的品牌或产品标识出现错字、乱码，或画面上出现大段乱码文字；画面里细小的
   次要文字（参数小字、界面小字、背景文字）看不清或糊掉不算
4. 严重崩坏：肢体扭曲、物体穿模、画面撕裂等一眼假

输出 json：
{"通过": true/false, "商品一致": true/false, "问题": "不通过时30字内说明，通过写空字符串"}
只输出 json。"""

# 装配/拆解序列的镜头，商品本来就是不完整的，第 1 条会把整块误拦掉，然后带着「把部件补回来」
# 的反馈重生成一次——正好和创意相反。所以清单必须跟着传到这里，把第 1 条反过来用。
# 只能按「全段从头到尾都不该出现」的部件判：一条序列通常整条落在一块里（_seg_blocks 会把连续
# 的补片镜头并成一次生成），块内某一帧该不该有屏幕，整段视频的验收模型分不出来，按每一步的
# 应无去判必然自相矛盾——第 1 步不许有屏幕，最后一步却必须有。
STATE_HINT = """
【这一块的商品是故意不完整的】__NOTES__
第 1 条对这一块反过来判：商品看起来是半成品、缺部件、缺印刷文字，都是对的，不要因此判不通过；
只有下面两种才算不合格：__NEVER__演变方向反了（一上来就是完整成品，或该装上的过程没发生）。
已经装上的那些部件仍然要与商品外观一致。
"""


def _state_hint(state_notes: dict) -> str:
    """把装配序列的演变说明渲染成成片验收里的例外条款；没有装配序列就返回空串。

    「不许出现」的清单分两种情况取：带「位置」说明这一块是一条演变过程（起始→结束），只能取
    块内各步应无的交集；不带「位置」说明这一步单独占一块，整段视频就是这一个形态，它自己的
    应无就是全段不许出现。混用两者会把「第 1 步不许有屏幕」加到整段上，而最后一步必须有屏幕。
    """
    rows, never = [], []
    for note in (state_notes or {}).values():
        if not isinstance(note, dict):          # 老任务的状态说明是纯字符串
            continue
        where = note.get("位置")
        if not (where or note.get("应无") or note.get("块内始终应无")):
            continue
        rows.append("%s%s：尚未装上 %s"
                    % (where + "状态" if where else "这一镜",
                       "（共 %s 步）" % note["共几步"] if where == "起始" and note.get("共几步")
                       else "", "、".join(note.get("应无") or []) or "（无，已是完整商品）"))
        for x in (note.get("块内始终应无") or []) if where else (note.get("应无") or []):
            if x not in never:
                never.append(x)
    if not rows:
        return ""
    return (STATE_HINT.replace("__NOTES__", "；".join(rows))
            .replace("__NEVER__", "画面里出现了 %s（哪怕只闪一帧、只露一角）；" % "、".join(never)
                     if never else ""))


def _vet_piece(rec: dict, video_url: str, part: dict, shots: dict, product: dict,
               state_notes: dict = None) -> dict:
    """生成块的成片验收：商品走样/画面无关/乱码大字/严重崩坏才拦，判不出放行。

    state_notes 带「应无」清单时（装配/拆解序列）验收口径反转：缺件不算走样，模型把缺的
    部件补全才算不合格——不传这份清单，这类创意会被第 1 条一路拦到底。
    结果按门禁范式（门禁/通过/使用）记进 piece，review_case 的 gates.violations
    自动对账「没过却被使用」，不用改复核器。"""
    brief = "\n".join("镜%s：%s；动作=%s" % (i, (shots.get(i) or {}).get("画面"),
                                              (shots.get(i) or {}).get("动作"))
                       for i in part.get("镜头序号") or [])
    try:
        raw = aigc.understand(PIECE_CHECK_PROMPT.replace("__SHOTS__", brief)
                              .replace("__NAME__", product.get("name") or "商品")
                              .replace("__LOOK__", product.get("appearance") or "")
                              .replace("__STATES__", _state_hint(state_notes)),
                              media=[{"type": "video", "url": video_url}],
                              max_tokens=512, json_mode=True)
        got = write_script._parse_json(raw)
        ok = rules.as_bool(got.get("通过"))
        return {"门禁": "片段验收", "通过": True if ok is None else ok,
                "商品一致": rules.as_bool(got.get("商品一致")),
                "原因": str(got.get("问题") or "")[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"门禁": "片段验收", "通过": True, "原因": "验收失败放行：%s" % str(exc)[:120]}


def _trim_to(path: str, need: float) -> float:
    """把补片裁回计划时长（seedance 最短出 4s，比计划短的块必然多出一截）。"""
    got = produce_video._duration(path)
    if need < 0.4 or got <= need + 0.25:
        return round(got, 2)
    tmp = os.path.splitext(path)[0] + "_trim.mp4"
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-i", path, "-t", "%.2f" % need, "-c:v", "libx264",
                             "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
                            + produce_video.aac_args() + [tmp])
    if ret.returncode != 0 or not os.path.isfile(tmp):
        return round(got, 2)          # 裁不动就留原长，节奏差一点也别丢整块
    os.replace(tmp, path)
    return round(produce_video._duration(path), 2)


def _gen_piece(rec: dict, seg: dict, block: dict, built: dict, product: dict,
               shots: dict, segdir: str, pool: dict = None, gap_facts: dict = None,
               model_url: str = "", voice_url: str = "", tag: str = "",
               label: str = "", cut_ok: bool = False, shot_refs: dict = None) -> dict:
    """补一块 AI 片：有可用的素材片段就做「素材编辑」，否则用参考图生视频按提示词还原。

    模特怎么出镜由 rules.py 的「切片缺失补片」表判定，音色参考由「音色优先级」表给出。
    整段都要补时 block 就是整段，此时与旧的整段生成完全等价。
    cut_ok=True 表示这些镜头的素材能直接出镜：参考视频被风控拒时直接抛出去让调用方改裁素材，
    不再退化成纯生成（纯生成等于把这条真素材整条丢掉）。
    """
    whole = len(block["镜头"]) == len(seg.get("镜头序号") or [])
    part = seg if whole else _sub_seg(seg, block, shots)
    facts = dict(gap_facts or {})
    facts["需要模特出镜"] = rules.as_bool(seg.get("需要模特出镜"))
    gap = rules.decide("切片缺失补片", facts)
    # 商品图按分镜计划取：这一块的镜头要什么角度/配色就带什么，没有计划才退回全片共用；
    # 中间状态图（视频模型一次画不准的关键状态，已用图像编辑锁死）排在最前面
    product_urls, ref_plan, state_notes = _piece_product_urls(product, shot_refs, block["镜头"])
    plan = _gap_refs(rec, part, built, product_urls, gap, model_url)
    refs, char_ids = plan["refs"], plan["char_ids"]
    product_urls = plan["product_urls"]      # 清单编号只认真正下发的那几张商品图

    clip, clip_src = _pick_ref_clip(rec, block["rows"], shots, pool or {}, tag, label)
    ref_videos = [storage.upload(clip)] if clip else []

    # 我们自己在 compose 里铺音轨/烧字幕，所以段内不要模型生成的音乐与画面文字（画面内音效保留）
    # 状态说明只留真正下发的那几张图：提示词与成片验收都按它写，两处口径必须一致
    notes = {u: s for u, s in state_notes.items() if u in product_urls}
    opt = produce_video.optimize_prompt(part, shots, char_ids, product_urls,
                                        built.get("素材") or {}, product, notes)
    body = opt["prompt"].strip()
    for kw in (produce_video.LEAD_NO_SUBTITLE, produce_video.LEAD_NO_BGM):
        while body.startswith(kw):
            body = body[len(kw):].lstrip()
    lead = produce_video.LEAD_NO_SUBTITLE + produce_video.LEAD_NO_BGM
    prompt = (lead + (EDIT_HINT if ref_videos else "")
              + (line_art.REAL_ACTOR_HINT if plan["线稿人物"] else "")
              + plan["extra"] + body)
    # 不能只把台词作为结构化信息交给 prompt 优化器：优化器可能保留台词
    # 文本，却没有把「需要模型生成对白人声」传到最终视频提示词。这里在
    # 真正下发前再加一道确定性的音频约束，保证提嗓逻辑与生成要求一致。
    dialogue = [str(_shot_text(shots.get(no)) or "").strip()
                for no in block["镜头"] if str(_shot_text(shots.get(no)) or "").strip()]
    if dialogue:
        prompt += ("\n【对白与人声】本段有明确对白。对应镜头中人物必须说出以下台词，"
                   "生成清晰自然、与口型同步的人声，不要只做默剧：%s"
                   % "；".join(dialogue))
    else:
        prompt += "\n【对白与人声】本段无台词，只保留必要的画面内物理音效，不生成对白或人声。"

    def _gen(p_text, videos, audios):
        # keep_refs：商品图不参与线稿降级——线稿化只保人物外观，商品图会被改造成人物图
        return line_art.gen_video_safe(p_text, ref_images=refs or None,
                                       keep_refs=product_urls,
                                       ref_videos=videos or None,
                                       ref_audios=audios or None,
                                       duration_sec=part["生成时长秒"])

    # 参考视频/参考音色都可能被单独拒（真人风控 / 音频格式），一次拒就丢掉那一样重试，
    # 别让整块失败——丢参考视频还能走参考图生视频，丢参考音色只是音色退化。
    audios = [voice_url] if voice_url else []
    if audios and not refs and not ref_videos:
        # seedance 不收「参考音是唯一参考输入」的请求（纯文生视频块没有商品图/人物图时会撞上），
        # 提前丢掉参考音色，别等它 41000000 整块失败
        audios = []
        log(rec, "  %s 这一块没有参考图/参考视频，参考音色不能单独送，改纯文生视频" % label)
    for attempt in range(3):
        try:
            out = _gen(prompt, ref_videos, audios)
            break
        except RuntimeError as exc:
            if audios and _audio_rejected(exc):
                log(rec, "  %s 参考音色被拒（音频格式），丢掉参考音色重生成" % label)
                audios = []
            elif ref_videos and _video_rejected(exc):
                if cut_ok:                # 素材能直接出镜就别退化成纯生成，交给调用方裁素材
                    raise RuntimeError("参考视频被真人风控拒") from exc
                log(rec, "  %s 参考视频被真人风控拒，丢掉参考视频改走参考图生视频" % label)
                ref_videos = []
                prompt = prompt.replace(EDIT_HINT, "", 1)
            else:
                raise
    file = storage.download(out["video_url"], os.path.join(segdir, "seg_%s.mp4" % tag))
    # 验收门禁：商品走样/画面无关/乱码/崩坏 → 把问题写进提示词重生一次；仍不过就取较好的
    # 一版并留下「通过=False 使用=True」记录，review_case 会对账出来
    vet = _vet_piece(rec, out["video_url"], part, shots, product, notes)
    if vet.get("通过") is False:
        log(rec, "  %s 片段验收不通过（%s），带反馈重生成一次" % (label, vet.get("原因")))
        try:
            p2 = prompt + "特别注意，上一次生成出现且必须避免：%s。" % (vet.get("原因") or "")
            out2 = _gen(p2, ref_videos, audios)
            f2 = storage.download(out2["video_url"],
                                  os.path.join(segdir, "seg_%s_v2.mp4" % tag))
            vet2 = _vet_piece(rec, out2["video_url"], part, shots, product, notes)
            if vet2.get("通过") is not False or (vet2.get("商品一致") is not False
                                                 and vet.get("商品一致") is False):
                out, file, vet, prompt = out2, f2, vet2, p2
                log(rec, "  %s 重生成后验收：%s" % (label,
                    "通过" if vet.get("通过") else "仍不通过，取较好一版"))
        except Exception as exc:  # noqa: BLE001  已有可用的第一版，重生失败不拖垮整块
            log(rec, "  %s 重生成失败，沿用第一版（%s）" % (label, str(exc)[:80]))
    vet["使用"] = True
    return {"kind": "gen", "mode": "edit_user_video" if ref_videos else out["mode"],
            "验收": vet,
            "file": file, "url": out["video_url"], "镜头": list(block["镜头"]),
            "prompt_final": prompt, "ref_images": refs, "ref_videos": ref_videos,
            # 溯源：ref_images_used 是真人风控降级后真正下发的参考图（人物图换成了线稿版），
            # 参考视频来源回答「@视频1 剪自哪条素材的哪几秒」，都进 segments.json 供报告用
            "ref_images_used": out.get("ref_images"),
            "参考视频文件": clip or "", "参考视频来源": clip_src,
            "ref_audios": audios, "补片判定": gap, "商品图计划": ref_plan,
            "prompt_fixed": opt.get("fixed"), "optimize_error": opt.get("optimize_error"),
            "生成时长秒": part["生成时长秒"],
            "时长秒": _trim_to(file, 0 if whole else float(part["原始时长秒"]))}


def _assemble_segments(rec: dict, built: dict, pieces: list, segdir: str) -> list:
    """把各片按段号归位、段内顺序拼成段视频，返回和以前同形的分段结果。

    段内混合（用户素材 + AI 补片）时两种来源的分辨率/帧率/编码都不一样，必须走
    produce_video.concat 的归一化拼接，不能直接复制流。
    """
    tid, results = rec["task_id"], []
    for seg in built["分段"]:
        group = sorted((p for p in pieces if p["段号"] == seg["段号"]), key=lambda p: p["序"])
        bad = [p for p in group if p.get("error")]
        if bad or not group:
            results.append({"段号": seg["段号"], "片": group,
                            "error": "；".join("%s %s" % (p["label"], p["error"]) for p in bad)
                                     or "这一段没有可用的片"})
            continue
        out = group[0]["file"]
        if len(group) > 1:
            out = os.path.join(segdir, "seg_%02d.mp4" % seg["段号"])
            mix = produce_video.concat([p["file"] for p in group],
                                       _d(tid, "generated", "mix%02d" % seg["段号"]))
            if mix.get("error"):
                results.append({"段号": seg["段号"], "片": group,
                                "error": "段内拼接失败：%s" % mix["error"]})
                continue
            os.replace(mix["file"], out)
        kinds = {p["kind"] for p in group}
        results.append({"段号": seg["段号"], "file": out, "片": group,
                        "mode": "mix_user_ai" if len(kinds) > 1 else group[0]["mode"],
                        "url": group[0].get("url") if len(group) == 1 else "",
                        "用户素材镜头": [n for p in group if p["kind"] == "cut" for n in p["镜头"]],
                        "AI补片镜头": [n for p in group if p["kind"] == "gen" for n in p["镜头"]],
                        "生成时长秒": round(produce_video._duration(out), 1)})
    return results


def _voice_probe_order(jobs: list, shots: dict) -> list:
    """要串行试「提嗓子」的 AI 补片块，按最可能出声排序。

    台词字数最多的块最可能真把话念出来（并列看计划时长），没台词的块提不出人声——
    按 jobs 原顺序从段1 试起，开场那种没台词的画面块每试一次就白花一整个生成时长。
    一块有台词的都没有时返回空列表：没有在提示词里要求对白的 AI 片段，
    不应该被拿来串行等待并提取音色。
    """
    voiced = []
    for job in jobs:
        if job["block"]["kind"] == "cut":
            continue
        text = "".join(_shot_text(shots.get(no)) or "" for no in job["block"]["镜头"])
        if text:
            voiced.append((len(text), _block_planned(job["block"], shots), job))
    return [c[2] for c in sorted(voiced, key=lambda c: (-c[0], -c[1]))]


# 音色探针提示词：这条片子唯一的用途是拿一段干净的人声，画面越简单越不容易被模型加戏。
# 光有台词不够——必须把「谁在说、用什么嗓子说」一起写清楚，否则模型可能给旁白、
# 换个性别的声音，甚至只动嘴不出声，提纯出来的音色和成片对不上。
PROBE_PROMPT = ("保持无字幕。无bgm。固定机位中近景，__WHO__正对镜头说话，"
                "光线均匀柔和，背景是干净的纯色墙面，除这一个人外画面里没有其他人或文字。\n"
                "【台词】人物必须完整、清晰地说出这一句，口型与声音严格同步：{__LINE__}\n"
                "【声音要求】__VOICE__，普通话，咬字清晰，语速正常，"
                "全程持续出声不留长时间静默，音量适中稳定；\n"
                "整条只有这一个人的说话声：不要旁白与画外音、不要背景音乐、不要环境音与音效、"
                "不要第二个人的声音、不要混响与回声、不要变声或电子处理，人声干净突出。")


def _probe_speaker(built: dict) -> "tuple[str, str]":
    """探针里念台词的人，返回（画面里的人, 声音要求）。

    音色基准要拿去克隆和当 seedance 参考音，所以性别年龄必须跟成片主角一致：
    外观字段开头就是「性别年龄」（见 write_script.WRITE_PROMPT 的 schema），
    性格字段写的是「性格与说话方式」，正好是声音的语气来源，两者都得下发。
    """
    chars = (built.get("剧本") or {}).get("人物设定") or []
    lead = next((c for c in chars if "主角" in str(c.get("角色") or "")), None) or (
        chars[0] if chars else {})
    look = str(lead.get("外观") or "").strip()
    tone = str(lead.get("性格") or "").strip()
    who = ("一位%s" % look[:80]) if look else "一位真人演员"
    voice = "真人原声，音色与这个人的性别年龄一致（%s）" % (look[:40] or "成年真人")
    if tone:
        voice += "，说话语气与其性格相符：%s" % tone[:40]
    return who, voice


# 探针只有 4s，塞不下长台词：模型会赶着念完（音色失真）或念一半被截断。
# 按 ≈4.5 字/秒估容量，在句读处截，既念得完整又留够 ≥2s 的可用人声（VOICE_BASE_MIN）。
PROBE_CPS = 4.5


def _probe_line(line: str, sec: int) -> str:
    """把台词裁到探针时长念得完的长度，优先在句读处断开。"""
    line = (line or "").strip()
    limit = max(8, int(sec * PROBE_CPS))
    if len(line) <= limit:
        return line
    cut = max(line.rfind(p, 0, limit + 1) for p in "，。！？；、…,.!?;")
    got = line[:cut + 1] if cut >= 8 else line[:limit]
    # 断在逗号/顿号上会留个尾巴（「…450SR，」），念起来像话没说完，去掉这类弱句读
    return got.strip().rstrip("，、；,;… ")


def _voice_probe_gen(rec: dict, built: dict, shots: dict, jobs: list, voice: dict) -> bool:
    """先用一条 4s 低分辨率片段专门取音色，成功后所有分段就能直接并发。

    正片片段又长又贵（单块最长 15s、还要过验收和重生成），只为了拿一把嗓子而串行等它，
    整步会被拖成「先跑完一整块，再开始并发」。探针只承担「出一句人声」这一件事：
    时长压到 seedance 下限 4s、分辨率压到 config.VOICE_PROBE_RESOLUTION，
    纯文生视频不带参考图（不碰真人风控降级），拿到人声就提纯成全片基准音。
    失败（网关不认这个分辨率、没出声、提纯太弱）返回 False，调用方回退到拿正片提音色。
    """
    order = _voice_probe_order(jobs, shots)
    if not order:
        return False
    block = order[0]["block"]
    line = _probe_line(_join_lines(_shot_text(shots.get(no)) for no in block["镜头"]),
                       config.VOICE_PROBE_SEC)
    if not line:
        return False
    who, voice_req = _probe_speaker(built)
    try:
        url = aigc.gen_video(PROBE_PROMPT.replace("__WHO__", who)
                             .replace("__VOICE__", voice_req)
                             .replace("__LINE__", line),
                             duration_sec=config.VOICE_PROBE_SEC,
                             resolution=config.VOICE_PROBE_RESOLUTION)
        file = storage.download(url, _p(rec["task_id"], "audio", "voice_probe.mp4"))
    except Exception as exc:  # noqa: BLE001  探针只是加速手段，失败不能影响出片
        log(rec, "音色探针生成失败，回退到用正片提音色：%s" % str(exc)[:150])
        return False
    ok = _voice_from_piece(rec, voice, {"file": file, "label": "音色探针"})
    log(rec, "音色探针（%ds/%s，台词「%s」）%s"
        % (config.VOICE_PROBE_SEC, config.VOICE_PROBE_RESOLUTION, line,
           "取到人声，全部分段并发出片" if ok else "没取到可用人声"))
    return ok


def step_generate(rec: dict) -> dict:
    tid = rec["task_id"]
    with open(_p(tid, "script", "script.json"), encoding="utf-8") as fh:
        built = json.load(fh)
    with open(_p(tid, "product", "fact_card.json"), encoding="utf-8") as fh:
        product = json.load(fh)
    with open(_p(tid, "edit", "asset_matches.json"), encoding="utf-8") as fh:
        matches = json.load(fh)["分镜匹配"]
    shots = {s.get("序号"): s for s in built["剧本"].get("分镜") or []}
    segdir = _d(tid, "generated", "segments")
    # 素材标注用来判断片段里有没有真人出镜（能不能做视频编辑）和有没有模特人脸（补片参考图）
    index_path = _p(tid, "assets", "material_index.json")
    pool = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fh:
            pool = {s.get("片段ID"): s for s in (json.load(fh).get("片段") or [])}

    # 「素材编辑」当不了参考视频时先降级成直接裁剪：用户素材为主，别掉到纯生成
    _demote_edit_rows(rec, matches, pool)

    # 「切片缺失补片」表要的两个全片级事实。
    # 没有任何选中的素材切片是确定事实（无模特出镜=False），不是「未判定」：
    # 记 None 会让规则表四行全落空掉进兜底「AIGC直接生成」，把人物设定图整个丢掉。
    used = [r for r in matches if r.get("策略") == "直接裁剪" and r.get("片段ID")]
    faces = [rules.as_bool((pool.get(r["片段ID"]) or {}).get("模特人脸出镜")) for r in used]
    gap_facts = {"已有模特出镜": False if not faces
                 else (True if any(f is True for f in faces)
                       else (False if all(f is False for f in faces) else None)),
                 "多段需要模特": sum(1 for s in built["分段"]
                                     if rules.as_bool(s.get("需要模特出镜")) is True) > 1}
    model_url = ""
    if gap_facts["已有模特出镜"] and any(rules.as_bool(s.get("需要模特出镜")) is True
                                        for s in built["分段"]):
        model_url = _model_frame_line_art(rec, used, pool)
    # 成片要不要口播是全片级事实，不看参考片有没有口播，看下游谁会用这把嗓子：
    # 要重配音的切片行、要 AI 补片的行，只有**这一镜真有台词**才算需要。`需要配音` 只说明
    # 这段素材的原声不能要（画面能用但口播对不上），这一镜没台词时 _dub_wav 照样跳过，
    # 白提一次音色基准，还会拖出下面「先串行跑一块提嗓子」的慢路径。
    # 一句台词都没有时成片音轨完全由参考片音轨策略决定（比如直接贴爆款 BGM），整条跳过。
    need_voice = any(_shot_text(shots.get(r.get("序号")))
                     for r in matches if r.get("需要配音") or not _can_cut(r))
    if need_voice:
        voice = _voice_plan(rec, matches, built["分段"], pool)
    else:
        voice = {"策略": "无需口播", "来源": "", "基准音文件": "", "基准音URL": "",
                 "说明": "没有需要配音的切片，AI 补片镜头也没有台词，跳过音色基准提取"}
        with open(_p(tid, "audio", "voice_plan.json"), "w", encoding="utf-8") as fh:
            json.dump(voice, fh, ensure_ascii=False, indent=2)
        log(rec, "音色：成片无口播（无配音切片、补片无台词），跳过音色基准提取")

    # 展平成「片」再并发：段内各片彼此独立，串着跑会让补片多的段拖长整步
    jobs = []
    for seg in built["分段"]:
        blocks = _seg_blocks(seg, _seg_matches(seg, matches))
        for i, blk in enumerate(blocks, 1):
            tag = "%02d" % seg["段号"] + ("_%d" % i if len(blocks) > 1 else "")
            jobs.append({"seg": seg, "block": blk, "序": i, "tag": tag,
                         "label": "段%d" % seg["段号"] + ("块%d" % i if len(blocks) > 1 else "")})

    alloc = _ClipWindows()   # 任务级窗口分配：同一片段多镜头复用时错开画面
    # 要补片的镜头才需要商品参考图，按剧本逐镜分配（卡点换装这类每拍要的图不一样）
    shot_refs = _plan_shot_refs(rec, built, product,
                                [no for j in jobs if j["block"]["kind"] != "cut"
                                 for no in j["block"]["镜头"]])
    # 视频模型一次画不准的关键状态先用图像编辑锁成一张中间图，再当参考图下发
    shot_refs = _make_state_images(rec, shot_refs, product)

    def one(job):
        seg, blk = job["seg"], job["block"]

        def cut(note=""):
            # 先合成念白再裁画面：TTS 语速比参考片慢，按计划时长裁完再贴必然砍半句话。
            # 念白装不下就先放长画面（多取素材里没用到的真秒数），实在不够才加速念白。
            dub = _dub_wav(rec, blk, shots, voice, job["tag"]) if any(
                r.get("需要配音") for r in blk["rows"]) else {}
            scale = _dub_scale(float(dub.get("dur") or 0), _block_planned(blk, shots))
            piece = _cut_piece(rec, blk, shots, segdir, job["tag"], scale, alloc)
            if dub.get("wav"):
                # 画面可能比计划还短（素材放慢到顶也补不齐），按实际长度再放长一次
                room = float(dub["dur"]) / DUB_MAX_TEMPO
                got = float(piece.get("时长秒") or 0)
                if room > got + 0.3:
                    piece = _cut_piece(rec, blk, shots, segdir, job["tag"],
                                       scale * room / max(0.4, got), alloc)
            if dub:
                piece.update(_paste_dub(rec, piece, dub))
            if note:
                piece["降级"] = note
            log(rec, "  %s 裁用户素材 镜%s（%.1fs%s）%s%s"
                % (job["label"], blk["镜头"], piece["时长秒"],
                   "，放长 %.2f×" % scale if scale > 1.005 else "",
                   piece.get("配音") or "原声保留", "，" + note if note else ""))
            return piece

        try:
            if blk["kind"] == "cut":
                piece = cut()
            else:
                # 补片走不通（风控/接口/额度）时，只要这些镜头的素材本来就能出镜就退回用素材：
                # 用户素材优先于 AI 重演，也顺带让整段不会因为一次生成失败而废掉。
                cut_ok = all(_cuttable(r) for r in blk["rows"])
                try:
                    piece = _gen_piece(rec, seg, blk, built, product, shots, segdir, pool,
                                       gap_facts, model_url, voice.get("基准音URL") or "",
                                       job["tag"], job["label"], cut_ok, shot_refs)
                    _fill_silent_dub(rec, piece, blk, shots, voice, job["tag"], job["label"])
                except Exception as exc:  # noqa: BLE001
                    if not cut_ok:
                        raise
                    piece = cut("补片走不通（%s）改用素材" % str(exc)[:60])
        except Exception as exc:  # noqa: BLE001
            piece = {"kind": blk["kind"], "镜头": list(blk["镜头"]), "error": str(exc)[:400]}
        piece.update({"段号": seg["段号"], "序": job["序"], "label": job["label"]})
        return piece

    workers = max(1, int(rec["options"].get("seg_workers") or 5))
    pieces, rest = [], list(jobs)
    probed = []          # 串行提音色时跑过的片：它们的 _fill_silent_dub 早于基准音就绪
    # 只有「本块确实有台词」的 AI 片段才可能成为音色探针。没有台词的
    # 片段即使偶然带出背景人声，也不是我们要求模型生成的口播，不能为了
    # 等它而把整条流水线串行阻塞。
    voice_probe_jobs = _voice_probe_order(jobs, shots)
    # 一键直出：只有一个 AI 块、且没有要静音重配的裁剪块时，这把嗓子只服务它自己，
    # 而模型本来就会用自己的嗓子念这句台词——探针和串行提取都是纯空转（≤15s 的全片
    # 只分一段一块，最常撞上这条）。素材里有人声时 _voice_plan 已经备好基准音，
    # 这块万一生成成哑片，块内的 _fill_silent_dub 照样能补。
    solo_gen = (len([j for j in jobs if j["block"]["kind"] != "cut"]) <= 1
                and not any(r.get("需要配音") and _shot_text(shots.get(r.get("序号")))
                            for j in jobs if j["block"]["kind"] == "cut"
                            for r in j["block"]["rows"]))
    if need_voice and not voice.get("基准音文件") and voice_probe_jobs and not solo_gen:
        # 先走探针：4s 低分辨率片段拿到人声后，下面所有块都能一起并发。
        # 探针不行（网关不认分辨率/没出声）才退回老路：串行跑最可能出声的正片块。
        if not _voice_probe_gen(rec, built, shots, jobs, voice):
            for job in voice_probe_jobs:
                rest.remove(job)
                pieces.append(one(job))
                probed.append((job, pieces[-1]))
                if _voice_from_piece(rec, voice, pieces[-1]):
                    break
    elif need_voice and not voice.get("基准音文件"):
        log(rec, "音色：%s，跳过探针与串行提取，直接并发出片"
            % ("没有带台词的 AI 补片" if not voice_probe_jobs
               else "只有一个 AI 块、且没有要重配音的裁剪块，这句口播由它自己念"))
    with cf.ThreadPoolExecutor(workers) as ex:
        pieces += list(ex.map(one, rest))
    # 串行提音色跑过的片，当时基准音还没到手（基准音正是从它们身上提的），块内那次
    # _fill_silent_dub 必然补不上。现在基准音就绪，给还哑着的补一次：TTS + ffmpeg，
    # 秒级，救回一句凭空断掉的口播。已经有声的片 _fill_silent_dub 自己会跳过。
    if voice.get("基准音文件"):
        for job, piece in probed:
            if piece.get("file"):
                _fill_silent_dub(rec, piece, job["block"], shots, voice,
                                 job["tag"], job["label"])
    results = _assemble_segments(rec, built, pieces, segdir)
    for r in results:
        log(rec, "  段%d %s（%s）" % (r["段号"], r.get("mode") or ("失败：" + r.get("error", "")[:80]),
                                     "、".join("%s %s" % (p["label"], p.get("mode") or "失败")
                                               for p in r.get("片") or [])))
    path = _p(tid, "generated", "segments.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    ok = [r for r in results if r.get("file")]
    if not ok:
        raise RuntimeError("所有分段都生成失败")
    if len(ok) < len(results):
        # 缺段就别往下走：拼出来的成片会静默少掉这些段的内容，看日志才发现
        bad = [r for r in results if not r.get("file")]
        raise RuntimeError("%d/%d 段生成失败，缺段的成片没有意义，先修掉再继续：%s"
                           % (len(bad), len(results),
                              "；".join("段%d %s" % (r["段号"], (r.get("error") or "")[:300])
                                        for r in bad)))
    cut_shots = sum(len(r.get("用户素材镜头") or []) for r in ok)
    gen_shots = sum(len(r.get("AI补片镜头") or []) for r in ok)
    log(rec, "镜头来源：用户素材 %d 镜 / AI 补片 %d 镜" % (cut_shots, gen_shots))
    return {"artifact": _rel(tid, path), "generated": len(ok), "planned": len(results),
            "modes": sorted({r.get("mode") for r in ok}),
            "user_shots": cut_shots, "ai_shots": gen_shots}
