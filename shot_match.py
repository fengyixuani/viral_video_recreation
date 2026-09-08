"""分镜 ↔ 用户素材匹配（步骤 match）。

LLM 给每个分镜挑候选片段，按匹配度分三路：直接裁剪 / 素材编辑 / 重新生成。
一个片段只服务一个分镜（rules.SEGMENT_EXCLUSIVE），抢不到就退次优候选，都被占才走 AIGC。
"""
import json
import os

import aigc  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import write_script  # pyright: ignore[reportImplicitRelativeImport]
from media import _sec
from task_store import _p, _rel, log


# ---------------- 步骤 6：分镜 ↔ 素材匹配 ----------------
MATCH_SYSTEM = "你是短视频剪辑师，负责把剧本分镜对到已有素材片段上。只输出 json。"

MATCH_PROMPT = """把下面每个分镜匹配到最合适的素材片段。

【剧本分镜】
__SHOTS__

【可用素材片段】
__SEGMENTS__

匹配规则（按重要性排序）：
1. 片段能表达这一镜的动作和叙事功能；
2. 需要露出商品的镜头，片段里商品要清晰可见；
3. 画面景别、机位运镜、场景氛围接近；
4. 片段时长够用（可裁剪，可轻微放慢，不能硬凑）；
5. 画面清晰稳定。

匹配度按 0-1 打分：0.8 以上表示可以直接用这段素材；0.55-0.8 表示画面可用但需要改造
（换商品、换背景、延长）；低于 0.55 表示不该用，宁可重新生成。
没有任何片段合适时 候选 写空数组 []。

一个片段最终只会被一个分镜使用（程序按匹配度分配，让位的分镜自动退到它的次优候选），
所以每个分镜要把**所有**真正可用的片段按匹配度从高到低列出来（最多 3 个），
不要因为「这个片段更适合前面某一镜」就少给候选。

另外每条候选必须给出两个布尔值（判不出来才写 null，不要写字符串）：
- 画面匹配：这段画面能不能拿来当这一镜用。**从宽判**：主体/商品对得上，能表达外观展示或
  功能展示、使用过程、效果反馈之类的叙事作用，且与这一镜台词说的事不冲突，就写 true
  （允许裁剪、放慢、局部改造，不要求逐镜复刻参考片）；明显是别的商品、别的场景，或画面与
  台词自相矛盾（台词说涂抹却在展示包装盒之类），才写 false；
- 口播内容匹配：素材片段里听到的人声内容与这一镜台词说的是不是同一件事（语义一致即可，
  不要求逐字相同）；这一镜没有台词、或素材里没有人声 → false。它**只决定这段的声音怎么处理**
  （对得上留原声，对不上静音重配），不影响画面是否采用。

输出 json：
{"匹配":[{"序号":1,"候选":[{"片段ID":"","匹配度":0.0,"口播内容匹配":false,"画面匹配":false,
  "理由":"一句话说明为什么这样匹配"}]}]}
每个分镜都要有一条，序号与剧本一致。只输出 json。"""


def _shot_brief(shot: dict) -> str:
    return ("镜%s：时长%ss；景别=%s；运镜=%s；画面=%s；动作=%s；台词=%s；叙事功能=%s"
            % (shot.get("序号"), shot.get("时长秒"), shot.get("景别"), shot.get("运镜"),
               shot.get("画面"), shot.get("动作"), shot.get("台词") or "无",
               shot.get("叙事功能")))


def _segment_brief(seg: dict) -> str:
    return ("%s：源=%s；%s→%s（%ss）；%s；限制=%s"
            % (seg.get("片段ID"), seg.get("视频") or os.path.basename(seg.get("源文件") or ""),
               seg.get("开始时间"), seg.get("结束时间"), seg.get("时长秒"),
               seg.get("召回文本"), seg.get("限制条件") or "无"))


def _strategy(score: float) -> str:
    """匹配度 → 出片策略，口径见 rules.py 的「素材匹配策略」表。"""
    return rules.decide("素材匹配策略", {"匹配度": score})["动作"]


# 动作名（rules.CLIP_RULES 的「动作」）→ 裁剪这条切片时怎么处理它的音轨。
# None = 这条切片不能用；其余是 cut_clip 之后对音轨做的动作，实现见 _clip_audio。
CLIP_ACTIONS = {"原声直接使用": "keep", "静音后使用": "mute",
                "分离BGM保留人声": "vocal_only", "不使用": None}


def _clip_facts(seg: dict, hit: dict) -> dict:
    """凑齐「用户切片处理」表要的事实：门槛来自匹配判定，两个维度来自素材标注。

    素材标注缺失时留 None（未判定），由规则表的通配吃掉，不在这里猜。
    """
    seg = seg or {}
    return {"口播内容匹配": rules.as_bool(hit.get("口播内容匹配")),
            "画面匹配": rules.as_bool(hit.get("画面匹配")),
            "真人出镜口播": rules.as_bool(seg.get("真人出镜口播")),
            "有BGM": rules.as_bool(seg.get("有BGM"))}


def _cand_score(cand: dict) -> float:
    """把候选里的匹配度收敛成 0~1 的浮点；写不出数字算 0。"""
    try:
        value = float((cand or {}).get("匹配度") or 0)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def _match_candidates(item: dict, by_id: dict) -> list:
    """一条匹配结果整理成按匹配度降序的候选列表（兼容模型只回单个片段的旧格式）。"""
    raw = (item or {}).get("候选")
    if not isinstance(raw, list):
        raw = [item] if (item or {}).get("片段ID") else []
    out = [dict(c, 匹配度=_cand_score(c)) for c in raw
           if isinstance(c, dict) and by_id.get(c.get("片段ID"))]
    out.sort(key=lambda c: -c["匹配度"])
    return out


def _no_key(no) -> int:
    """分镜序号转成排序用的 int。

    序号是模型产出的字段：漏写时 str(None) → int("None") 抛 ValueError，写成「第1镜」
    同样抛——而这里只是排序的 tie-break，没理由让整个 match 步骤崩掉。认不出的排到最后。
    """
    try:
        return int(str(no).strip())
    except (TypeError, ValueError):
        return 1 << 30


def _assign_segments(shots: list, matched: dict, by_id: dict) -> dict:
    """给分镜派素材片段：**一个片段只服务一个分镜**（口径见 rules.SEGMENT_EXCLUSIVE）。

    允许跨分镜复用时，同一片段被两镜命中，第二刀已经没有新鲜画面，只能回卷复用起点，
    成片同画面播两遍（实测 #0004 只有 3.0s，镜8/镜9 各要 3.7s 就是这么重复的）。
    改成独占分配：匹配度高的镜先占，让位的镜退到自己的次优候选，候选都被占（或压根没有
    过门槛的候选）才退回 AIGC 生成。返回 {分镜下标: {"候选": 选中的那条或 None, "说明": 痕迹}}。

    键用**下标**而不是序号：模型给出两个相同序号时（写剧本那侧只排序、不去重），
    按序号做键会让两镜塌缩成一条、拿到同一个片段ID，正好复现上面要消灭的「同画面播两遍」。
    """
    cands = {i: _match_candidates(matched.get(str(s.get("序号"))), by_id)
             for i, s in enumerate(shots)}
    ranked = sorted(((c["匹配度"], _no_key(shots[i].get("序号")), rank, i, c)
                     for i, lst in cands.items() for rank, c in enumerate(lst)
                     if c["匹配度"] >= rules.EDIT_SCORE),
                    key=lambda x: (-x[0], x[1], x[2]))
    chosen, taken = {}, {}
    for _, _, rank, i, cand in ranked:
        if i in chosen or cand["片段ID"] in taken:
            continue
        note = ""
        if rank:
            first = cands[i][0]["片段ID"]
            holder = shots[taken[first]].get("序号") if first in taken else "?"
            note = ("首选片段 %s 已分给镜%s，改用次优候选 %s（一个片段只服务一个分镜）"
                    % (first, holder, cand["片段ID"]))
        chosen[i] = {"候选": cand, "说明": note}
        taken[cand["片段ID"]] = i
    for i, lst in cands.items():
        over = [c["片段ID"] for c in lst if c["匹配度"] >= rules.EDIT_SCORE]
        if i not in chosen and over:
            chosen[i] = {"候选": None,
                         "说明": "候选片段 %s 都已分给其它分镜，改走重新生成"
                                 "（一个片段只服务一个分镜）" % "、".join(over[:3])}
    return chosen


CLIP_USE_ACTIONS = ("原声直接使用", "分离BGM保留人声", "静音后使用")


def _can_cut(row) -> bool:
    """这个镜头能不能直接裁用户素材出镜。"""
    return bool(row and row.get("策略") == "直接裁剪" and row.get("源文件")
                and (row.get("切片动作") or "原声直接使用") in CLIP_USE_ACTIONS)


def _cuttable(row) -> bool:
    """画面过了门槛、素材也在手上——不管当前策略是什么，这个镜头都能直接裁出镜。"""
    return bool(row and row.get("源文件")
                and (row.get("切片动作") or "") in CLIP_USE_ACTIONS
                and float(row.get("可用秒") or 0) > 0)


def _shot_text(shot: dict) -> str:
    """分镜台词的裸念白（去掉「角色：」前缀，「无」当没有台词）。"""
    text = str((shot or {}).get("台词") or "").strip()
    text = text.split("：", 1)[-1].strip() if text else ""
    return "" if text in ("", "无", "-") else text


def step_match(rec: dict) -> dict:
    tid = rec["task_id"]
    with open(_p(tid, "script", "script.json"), encoding="utf-8") as fh:
        script = json.load(fh)
    with open(_p(tid, "assets", "material_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    shots = script["剧本"].get("分镜") or []
    pool = index.get("片段") or []
    result = {"启用素材": bool(pool), "分镜匹配": []}

    matched = {}
    if pool:
        raw = aigc.understand(
            MATCH_PROMPT.replace("__SHOTS__", "\n".join(_shot_brief(s) for s in shots))
                        .replace("__SEGMENTS__", "\n".join(_segment_brief(s) for s in pool)),
            system=MATCH_SYSTEM, max_tokens=8192, json_mode=True)
        for item in write_script._parse_json(raw).get("匹配") or []:
            matched[str(item.get("序号"))] = item

    by_id = {s.get("片段ID"): s for s in pool}
    assign = _assign_segments(shots, matched, by_id) if pool else {}
    for index, shot in enumerate(shots):
        got = assign.get(index) or {}
        hit = got.get("候选") or {}
        seg = by_id.get(hit.get("片段ID"))
        # 先取整再判策略：产物里记的是 round(匹配度, 2)，用原值判会出现
        # 「匹配度 0.80 却走重新生成」这种产物自相矛盾（阈值区间见 rules.MATCH_RULES）。
        score = round(_cand_score(hit), 2) if seg else 0.0
        # 「用户切片处理」表先判这条切片能不能原样出镜（门槛 + 音轨怎么处理）。
        # 判成「不使用」的不做直接裁剪，但画面够用时仍可当素材编辑的参考视频。
        clip = rules.decide("用户切片处理", _clip_facts(seg, hit)) if seg else None
        strategy = _strategy(score)
        if clip and clip["动作"] == "不使用" and strategy == "直接裁剪":
            strategy = "素材编辑"
        row = {"序号": shot.get("序号"), "时长秒": shot.get("时长秒"),
               "片段ID": (seg or {}).get("片段ID", ""), "匹配度": score,
               "策略": strategy, "理由": hit.get("理由") or ""}
        if got.get("说明"):
            row["分配说明"] = got["说明"]
        if clip:
            row.update({"切片动作": clip["动作"], "需要配音": clip.get("需要配音"),
                        "切片判定": clip})
        if seg:
            row.update({"源文件": seg.get("源文件"), "开始秒": round(_sec(seg.get("开始时间")), 2),
                        "可用秒": round(_sec(seg.get("时长秒")), 2)})
        result["分镜匹配"].append(row)

    stat, clip_stat = {}, {}
    for row in result["分镜匹配"]:
        stat[row["策略"]] = stat.get(row["策略"], 0) + 1
        if row.get("切片动作"):
            clip_stat[row["切片动作"]] = clip_stat.get(row["切片动作"], 0) + 1
    result.update({"统计": stat, "切片处理统计": clip_stat})
    yields = [r for r in result["分镜匹配"] if r.get("分配说明")]
    if yields:
        result["片段独占让位"] = [{"序号": r["序号"], "说明": r["分配说明"]} for r in yields]
    path = _p(tid, "edit", "asset_matches.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log(rec, "素材匹配：%s" % ("、".join("%s%d镜" % (k, v) for k, v in stat.items()) or "无分镜"))
    if clip_stat:
        log(rec, "切片处理：%s" % "、".join("%s%d条" % (k, v) for k, v in clip_stat.items()))
    for row in yields:
        log(rec, "  镜%s %s" % (row["序号"], row["分配说明"]))
    return {"artifact": _rel(tid, path), "stat": stat, "clip_stat": clip_stat,
            "片段独占让位": len(yields)}
