"""单个任务的成片自检：素材利用率、逐片状态、字幕一致性、音轨连续性。

用法: python3 review_case.py <task_id> [task_id...]
只读产物 + ffmpeg 量音量，不调任何模型；给多轮测试提供同一把尺子。
"""
import json
import os
import re
import sys

import gates
import pipeline
import produce_video


def _read(task_id, *parts):
    path = pipeline._p(task_id, *parts)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _ass_cues(path):
    """从烧录用的 ASS 里读回 [(start, end, text)]，用来检查重复与交叠。"""
    if not os.path.isfile(path):
        return []
    def sec(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    cues = []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("Dialogue:"):
            continue
        cols = line.split(",", 9)
        if len(cols) < 10:
            continue
        text = re.sub(r"\{[^}]*\}", "", cols[9]).replace("\\N", " ").strip()
        cues.append((sec(cols[1].strip()), sec(cols[2].strip()), text))
    return sorted(cues)


def review(task_id):
    rec = pipeline.load(task_id)
    built = _read(task_id, "script", "script.json") or {}
    res = _read(task_id, "generated", "segments.json") or []
    report = {"task_id": task_id, "status": rec.get("status"),
              "参考片": os.path.basename(rec["inputs"].get("reference_video") or ""),
              "商品": (rec.get("product") or {}).get("name"),
              "素材条数": len(rec["inputs"].get("user_videos") or []),
              "问题": []}
    shots = built.get("剧本", {}).get("分镜") or []
    user_shots = sorted(n for r in res for n in (r.get("用户素材镜头") or []))
    ai_shots = sorted(n for r in res for n in (r.get("AI补片镜头") or []))
    report.update({"分镜数": len(shots), "用户素材镜头": user_shots, "AI补片镜头": ai_shots,
                   "素材利用率": round(len(user_shots) / max(1, len(shots)), 3)})
    # 0 镜用户素材要分两种：素材库本来就没有能出镜的片段（合理），
    # 还是有能裁的素材却被淘汰了（这才是缺陷——用户素材优先是产品原则）
    matches = (_read(task_id, "edit", "asset_matches.json") or {}).get("分镜匹配") or []
    cuttable = [r.get("序号") for r in matches if pipeline._cuttable(r)]
    report["可出镜素材镜"] = len(cuttable)
    if not user_shots and cuttable:
        report["问题"].append("有 %d 镜素材能直接出镜却 0 镜用户素材" % len(cuttable))
    elif not user_shots:
        report["素材可用性"] = ("素材库没有能出镜的片段（画面不匹配或全部停用），"
                              "全片 AI 生成属预期")

    # 同一素材片段的取用窗口重叠 = 成片同画面出现两次（显式标记过回卷的除外）
    wins, rolls = [], 0
    for r in res:
        for p in r.get("片") or []:
            for d in p.get("片段") or []:
                if d.get("复用回卷"):
                    rolls += 1
                elif d.get("源窗口"):
                    wins.append((d.get("片段ID") or "", float(d["源窗口"][0]),
                                 float(d["源窗口"][1]), r.get("段号")))
    wins.sort()
    lap = ["%s：段%s[%.1f-%.1f] 与 段%s[%.1f-%.1f]"
           % (a[0], a[3], a[1], a[2], b[3], b[1], b[2])
           for a, b in zip(wins, wins[1:]) if a[0] == b[0] and b[1] < a[2] - 0.5]
    if lap:
        report["问题"].append("素材窗口重叠（同画面重复出现）：%s" % "；".join(lap[:3]))
    if rolls:
        report["窗口回卷次数"] = rolls
        report["问题"].append("素材窗口回卷复用 %d 次（同画面重复出现，片段应当一镜独占）"
                            % rolls)

    final = pipeline._p(task_id, "render", "final.mp4")
    if not os.path.isfile(final):
        report["问题"].append("没有成片 final.mp4")
        return report
    report["成片秒"] = round(produce_video._duration(final), 2)
    planned = sum(max(0.1, float(s.get("时长秒") or 0)) for s in shots)
    report["剧本计划秒"] = round(planned, 2)
    if planned and abs(report["成片秒"] - planned) / planned > 0.25:
        report["问题"].append("成片时长与剧本计划差 >25%%（%.1fs vs %.1fs）"
                              % (report["成片秒"], planned))

    pieces, quiet = [], []
    lines = {s.get("序号"): (s.get("台词") or "").strip() for s in shots}
    for p in pipeline._pieces(built, res):
        err = produce_video._run([pipeline._ffmpeg(), "-hide_banner",
                                  "-ss", "%.2f" % p["start"],
                                  "-t", "%.2f" % (p["end"] - p["start"]), "-i", final,
                                  "-vn", "-af", "volumedetect", "-f", "null", "-"]).stderr
        mean = next((float(l.split("mean_volume:")[1].split("dB")[0])
                     for l in err.splitlines() if "mean_volume:" in l), 0.0)
        talk = any(lines.get(n) for n in p["镜头"])
        pieces.append({"key": p["key"], "镜头": p["镜头"], "有台词": talk,
                       "start": round(p["start"], 2), "end": round(p["end"], 2),
                       "mean_db": mean})
        # 没台词的镜静音是分镜设计（快切/空镜），只有该说话的片没声才是缺陷
        if mean < gates.SILENT_DB and talk:
            quiet.append(p["key"])
    report["片"] = pieces
    if quiet:
        report["问题"].append("这些片该有口播却几乎没声：%s" % "、".join(quiet))

    dub_cut = [(p.get("label"), p.get("配音截断秒")) for r in res for p in r.get("片") or []
               if p.get("配音截断秒")]
    if dub_cut:
        report["配音截断"] = dub_cut
        report["问题"].append("配音被截断：%s" % "、".join("%s %.1fs" % x for x in dub_cut))
    stretch = [(p.get("label"), p.get("放长倍数")) for r in res for p in r.get("片") or []
               if p.get("放长倍数")]
    if stretch:
        report["画面放长"] = stretch
    downgrade = [(p.get("label"), p.get("降级")) for r in res for p in r.get("片") or []
                 if p.get("降级")]
    if downgrade:
        report["降级"] = downgrade

    subs = (rec.get("result") or {}).get("subtitles") or {}
    report["字幕"] = {k: subs.get(k) for k in ("mode", "burned", "blocks", "lines")}
    # 只读本轮真正烧进去的那份 ASS：判"不烧字幕"时 render 里可能还躺着上一轮的孤儿文件
    # （实测 17_creative 关字幕重跑后 capwork/captions_clone.ass 仍在），拿它对账会误判
    cues = []
    if subs.get("burned"):
        cues = _ass_cues(pipeline._p(task_id, "render", "capwork", "captions_clone.ass")) \
            if subs.get("mode") == "风格克隆" \
            else _ass_cues(pipeline._p(task_id, "render", "subtitles.ass"))
    report["字幕块"] = len(cues)
    # 同一句话隔得远是文案本身在重复（口播里常见地反复念商品名），只有挨在一起才是烧重了
    # 剧本台词本来就重复的不算烧重（军营连喊两镜「必胜！」这类复读是设计）：
    # 归一化后统计该句在剧本台词里的出现次数，字幕里出现得不比它多就是按剧本走
    def _norm(t):
        return re.sub(r"[\s，。！？!?、,.:：;；·…~〜\-]+", "", t or "")

    script_text = "".join(_norm(x.get("台词")) for x in shots)

    def _planned(t):
        n = _norm(t)
        return n and script_text.count(n) >= sum(1 for c in cues if _norm(c[2]) == n)

    dup = ["%s（%.1fs 与 %.1fs）" % (a[2], a[0], b[0])
           for i, a in enumerate(cues) for b in cues[i + 1:]
           if a[2] and a[2] == b[2] and b[0] - a[1] < 2.5 and not _planned(a[2])]
    over = ["%.1f-%.1f 与 %.1f-%.1f" % (a[0], a[1], b[0], b[1])
            for a, b in zip(cues, cues[1:]) if b[0] < a[1] - 0.05]
    tail = [c for c in cues if c[1] > report["成片秒"] + 0.15]
    if dup:
        report["问题"].append("字幕重复：%s" % "、".join(sorted(set(dup))[:5]))
    if over:
        report["问题"].append("字幕交叠：%s" % "、".join(over[:5]))
    if tail:
        report["问题"].append("字幕越过片尾 %d 条" % len(tail))
    if subs.get("burned") and not cues:
        report["问题"].append("标记已烧字幕但读不到 ASS")

    # 字幕事实链对账：参考片有字幕 + 剧本有台词/花字 → 成片就该有字幕；反过来不该有的也要揪。
    # 只看"事实 vs 成片"，不看配置——配置被显式关掉恰恰是该暴露的那类问题（26 号就是这么丢的）
    verdict = subs.get("判定") or {}
    want_text = sum(1 for x in shots if (x.get("台词") or "").strip()
                    or (x.get("花字") or "").strip())
    ref_caps = ((verdict.get("事实") or {}).get("参考片有字幕"))
    if ref_caps and want_text and not subs.get("burned"):
        report["问题"].append("参考片有字幕、剧本有 %d 镜文字，成片却没烧字幕（判定：%s）"
                              % (want_text, verdict.get("动作") or "未记录"))
    if cues and not want_text:
        report["问题"].append("剧本一个字都没有，成片却烧了 %d 条字幕" % len(cues))

    # 剧本层商品状态门禁对账：有违规没修掉、或审查根本没跑成，都是缺陷（老任务没有该字段，不误报）
    audit = (built.get("剧本") or {}).get("商品状态审查") or {}
    if audit:
        report["商品状态审查"] = audit.get("状态")
    if audit.get("未修复"):
        report["问题"].append("剧本商品状态审查有 %d 处违规未修复" % len(audit["未修复"]))
    if str(audit.get("状态") or "").startswith("审查失败"):
        report["问题"].append("剧本商品状态审查未执行成功（%s）" % audit.get("状态"))

    # 整片叙事连贯性对账：逐镜门禁全绿、整片却各讲各的，只有这一层反映得出来
    # （老任务没有该字段，不误报）
    tale = (built.get("剧本") or {}).get("叙事连贯审查") or {}
    if tale:
        report["叙事连贯审查"] = "%s（可理解性 %s）" % (tale.get("状态"),
                                                       tale.get("可理解性", "-"))
    if tale.get("状态") == "可理解性不足":
        report["问题"].append("整片叙事看不懂：可理解性 %s 分，盲测看成「%s」"
                              % (tale.get("可理解性", "-"), tale.get("盲测概要") or "-"))
    if str(tale.get("状态") or "").startswith("审查失败"):
        report["问题"].append("整片叙事连贯审查未执行成功（%s）" % tale.get("状态"))

    # 门禁对账（实测门禁范式，见 gates.py）：产物里「通过=False 却 使用=True」都是缺陷；
    # 新加门禁自动被这里覆盖，不用改复核器
    audited = []
    for parts in (("audio", "voice_plan.json"), ("audio", "reference_audio.json"),
                  ("generated", "segments.json")):
        got = _read(task_id, *parts)
        if got:
            audited += gates.violations(got, "/".join(parts))
    if audited:
        report["问题"].append("门禁违规使用：%s" % "；".join(audited[:3]))
    return report


def main(argv):
    for task_id in argv or []:
        r = review(task_id)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        print("=" * 20, task_id, "问题 %d 项" % len(r["问题"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
