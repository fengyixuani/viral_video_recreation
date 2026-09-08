"""音色基准与配音：让 AI 补片和用户素材保持同一把嗓子。

音色优先级见 rules 的「音色优先级」表：能带走的用户原声 > 分离出的纯人声 > AI 段提纯。
配音走 Agent_tools 的声音克隆，长度对不上时按 DUB_MAX_TEMPO / DUB_MAX_STRETCH 折中。
"""
import json
import os
import threading

import gates  # pyright: ignore[reportImplicitRelativeImport]
import produce_video  # pyright: ignore[reportImplicitRelativeImport]
import rules  # pyright: ignore[reportImplicitRelativeImport]
import storage  # pyright: ignore[reportImplicitRelativeImport]
from media import _ffmpeg, _sec
from ref_audio import separate_bgm
from shot_match import _can_cut, _shot_text
from task_store import _p, log

from Agent_tools.registry import tts_clone  # noqa: E402


# ---------------- 音色（rules.VOICE_PRIORITY）----------------
VOICE_KEEP_ACTIONS = ("原声直接使用", "分离BGM保留人声")   # 这些切片带得走用户原声
VOICE_BASE_MAX = 12.0        # 基准原声最多取多长，够克隆/参考就行
VOICE_BASE_MIN = 2.0         # seedance 参考音单段下限 2s，短于它整条请求会被拒


def _voice_candidates(keep: list, pool: dict):
    """按优先级逐个给出基准原声候选：{"源文件","开始秒","可用秒","片段ID","说明"}。

    先用成片里「能带走原声」的切片；然后素材里的真人口播片段——画面可能没被选中、
    或者被静音后使用，但音轨里那把嗓子就是用户本人，克隆音色够用；再退到旁白/画外音；
    最后实测音量兜底。做成生成器而不是单选：标注说有人声不代表真有，
    _voice_plan 逐个实测响度，太弱就换下一个（4496：标注有人声、实测 -42.6 dB 底噪）。
    """
    if keep:
        best = max(keep, key=lambda r: float(r.get("可用秒") or 0))
        yield {"源文件": best.get("源文件"), "开始秒": float(best.get("开始秒") or 0),
               "可用秒": float(best.get("可用秒") or 0), "片段ID": best.get("片段ID"),
               "说明": "取自成片里保留原声的切片"}
    for fact, note in (("真人出镜口播", "取自素材里的真人口播片段（画面未必出镜，只借音色）"),
                       ("有人声", "取自素材里有人声的片段（旁白/画外音，只借音色）")):
        tier = [s for s in (pool or {}).values()
                if rules.as_bool(s.get(fact)) is True and s.get("源文件")]
        for s in sorted(tier, key=lambda s: -_sec(s.get("时长秒")))[:3]:
            yield {"源文件": s.get("源文件"), "开始秒": _sec(s.get("开始时间")),
                   "可用秒": _sec(s.get("时长秒")), "片段ID": s.get("片段ID"), "说明": note}
    got = _voice_by_loudness(pool or {})
    if got:
        yield got


# 「声音台词」标注里出现这些说法就说明这一段没人声，别拿它当克隆基准（会克隆出噪音）
NO_VOICE_WORDS = ("无人声", "没有人声", "无台词", "无对话", "无语音")

VOICE_MIN_DB = gates.VOICE_MIN_DB  # 克隆基准音响度门槛唯一来源在 gates.py，此处只留别名


def _voice_by_loudness(pool: dict) -> dict:
    """没有任何切片标了「真人出镜口播」时，靠实测音量挑一条有人声的素材当克隆基准。

    标注可能缺（旧素材索引、或大模型漏判这一个字段），但「全片没有配音」是不能接受的结果：
    所有静音后使用的切片都会变成有台词却没声音的片。所以这里退到可测事实——
    先用「声音台词」标注排掉明确说没人声的，再实测平均音量取最响的一条。
    """
    cands = [s for s in pool.values()
             if s.get("源文件") and _sec(s.get("时长秒")) >= VOICE_BASE_MIN
             and not any(w in str(s.get("声音台词") or "") for w in NO_VOICE_WORDS)]
    cands.sort(key=lambda s: -_sec(s.get("时长秒")))
    for seg in cands[:8]:                      # 只探前 8 条，每条一次 volumedetect
        ret = produce_video._run([_ffmpeg(), "-hide_banner",
                                  "-ss", "%.2f" % _sec(seg.get("开始时间")),
                                  "-t", "%.2f" % min(VOICE_BASE_MAX, _sec(seg.get("时长秒"))),
                                  "-i", seg["源文件"], "-vn", "-af", "volumedetect",
                                  "-f", "null", "-"])
        db = next((float(l.split("mean_volume:")[1].split("dB")[0])
                   for l in ret.stderr.splitlines() if "mean_volume:" in l), -99.0)
        if db > VOICE_MIN_DB:
            return {"源文件": seg.get("源文件"), "开始秒": _sec(seg.get("开始时间")),
                    "可用秒": _sec(seg.get("时长秒")), "片段ID": seg.get("片段ID"),
                    "说明": "素材没标真人口播，实测音量挑出的有声片段（%.0f dB，只借音色）" % db}
    return {}


def _purify_voice_base(rec: dict, wav: str, src: dict, pool: dict) -> dict:
    """基准原声提纯：来源切片带 BGM（或标注缺失）就分离出纯人声再当基准。

    克隆/参考对参考音很敏感，混着 BGM 的基准会把伴奏「克」进配音里（表二
    「用户视频-原声」的口径就是纯净人声）。分离失败或分离后达不到克隆门槛（说明人声
    被一起滤掉了）就保留原样，原因写进 voice_plan，不静默。

    判「提纯是不是白干了」必须用 VOICE_MIN_DB 而不是更松的 SILENT_DB：
    调用方紧接着就拿 gates.check("克隆基准音") 按 VOICE_MIN_DB 卡这个文件，
    响度落在 (SILENT_DB, VOICE_MIN_DB] 这一段时，这里认为分离成功、os.replace 覆盖掉
    原始的混音基准（原始可能有 -20dB、完全够用），下一行门禁又按 -38 判它「近乎无声」
    把整条候选弃用——一个本来可用的第一优先级候选被自己的提纯步骤毁掉，且不可回退。
    """
    facts = pool.get(src.get("片段ID")) or {}
    if rules.as_bool(facts.get("有BGM")) is False:
        return {"基准音处理": "原声直取（标注无BGM）"}
    try:
        got = separate_bgm(wav, os.path.splitext(wav)[0] + "_vocal.m4a", want="vocal")
        pure = os.path.splitext(wav)[0] + "_vocal.wav"
        ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", got["file"], "-ac", "1", "-ar", "44100",
                                 "-c:a", "pcm_s16le", pure])
        if ret.returncode != 0 or not os.path.isfile(pure):
            return {"基准音处理": "分离后转码失败，保留原声：%s" % ret.stderr[-120:]}
        pure_db, raw_db = _mean_db(pure), _mean_db(wav)
        if pure_db <= VOICE_MIN_DB and pure_db < raw_db:
            return {"基准音处理": "分离后 %.1f dB 达不到克隆门槛 %.0f（人声被一起滤掉），"
                                  "保留原声 %.1f dB" % (pure_db, VOICE_MIN_DB, raw_db)}
        os.replace(pure, wav)
        return {"基准音处理": "已分离纯人声（%s，%.1f dB）" % (got["method"], pure_db)}
    except Exception as exc:  # noqa: BLE001
        return {"基准音处理": "分离失败，保留原声：%s" % str(exc)[:120]}


def _voice_from_piece(rec: dict, plan: dict, piece: dict) -> bool:
    """从第一个出声的 AI 补片里提纯人声当全片音色基准（表二「AI视频片段」的落地）。

    成功返回 True 并就地更新 plan（基准音文件/URL、voice_plan.json），后续块的克隆
    配音与 seedance 参考音都锚到它；这一片没声/提纯失败返回 False，调用方试下一块。
    """
    file = piece.get("file")
    if not file or not os.path.isfile(file) or _mean_db(file) <= SILENT_DB:
        return False
    tid = rec["task_id"]
    try:
        got = separate_bgm(file, _p(tid, "audio", "ai_voice_vocal.m4a"), want="vocal")
        dst = _p(tid, "audio", "voice_base.wav")
        ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", got["file"], "-t", "%.2f" % VOICE_BASE_MAX,
                                 "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", dst])
        gate = (gates.check("克隆基准音", dst) if ret.returncode == 0
                else {"门禁": "克隆基准音", "通过": False, "原因": "提纯转码失败"})
        if not gates.ok(gate):
            log(rec, "  %s 出声但提纯后人声过弱，试下一块" % piece.get("label"))
            return False
        plan.update({"基准音文件": dst, "基准来源片段": piece.get("label"),
                     "基准来源说明": "取自第一个出声的 AI 补片（提纯人声）",
                     "基准音处理": "已分离纯人声（%s）" % got["method"],
                     "基准音门禁": dict(gate, 使用=True)})
        try:
            plan["基准音URL"] = storage.upload(dst)
        except Exception as exc:  # noqa: BLE001
            plan["上传失败"] = str(exc)[:150]
        with open(_p(tid, "audio", "voice_plan.json"), "w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        log(rec, "音色基准：%s（%s）" % (plan["基准来源说明"], piece.get("label")))
        return True
    except Exception as exc:  # noqa: BLE001
        log(rec, "  %s 提纯人声异常：%s" % (piece.get("label"), str(exc)[:120]))
        return False


def _voice_plan(rec: dict, matches: list, segments: list, pool: dict = None) -> dict:
    """按「音色优先级」表定全片音色基准，并把基准原声抽成文件。

    产物 audio/voice_plan.json：来源、策略、基准音文件与 URL、判定痕迹。
    """
    tid = rec["task_id"]
    keep = [r for r in matches if _can_cut(r) and r.get("切片动作") in VOICE_KEEP_ACTIONS]
    voiced_pool = any(rules.as_bool(s.get("真人出镜口播")) is True
                      or rules.as_bool(s.get("有人声")) is True
                      for s in (pool or {}).values())
    available = []
    if keep or voiced_pool:
        # 「用户视频-原声」= 素材里有人声可当基准（保留原声的切片，或只借音色的片段），
        # 与 _voice_candidates 的搜索口径一致——否则判定说 AI、提取却拿素材，自相矛盾
        available.append("用户视频-原声")
    if any(not _can_cut(r) for r in matches):        # 有镜头要 AI 补片
        available.append("AI视频片段")
    if any(_can_cut(r) and r.get("切片动作") == "静音后使用" for r in matches):
        available.append("用户视频-静音")
    plan = rules.pick_voice(available)
    plan.update({"基准音文件": "", "基准音URL": ""})
    # 必须是 wav/mp3：seedance 的 reference_audio 只收这两种，m4a/aac 会被
    # 41000000「audio format ... is not valid ... in r2v」整条拒掉（实测每段必失败）。
    # 时长也有下限，单段参考音要求 2~15s。
    dst, tried, rejected = _p(tid, "audio", "voice_base.wav"), set(), []
    for src in _voice_candidates(keep, pool or {}):
        if not src.get("源文件") or src.get("片段ID") in tried:
            continue
        tried.add(src.get("片段ID"))
        take = min(VOICE_BASE_MAX, max(VOICE_BASE_MIN, float(src.get("可用秒") or 0)))
        ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                                 "-ss", "%.2f" % float(src.get("开始秒") or 0),
                                 "-t", "%.2f" % take, "-i", src["源文件"], "-vn",
                                 "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", dst])
        if ret.returncode != 0 or not os.path.isfile(dst):
            rejected.append({"片段": src.get("片段ID"), "原因": "抽取失败：%s" % ret.stderr[-100:]})
            continue
        # 真实产出时长也要卡下限：take 只是 ffmpeg 的 -t，素材本身只有 0.8s 时输出还是 0.8s。
        # 门禁只量响度不量长度，于是这条过短的基准音会被上传成 seedance 的 reference_audio，
        # 每个 AI 块都先撞一次 41000000 再靠 _audio_rejected 丢参考音重生成（白花额度 +
        # 音色退化），本地克隆也拿到一段过短的 prompt。
        got_sec = produce_video._duration(dst)
        if got_sec < VOICE_BASE_MIN - 0.05:
            rejected.append({"片段": src.get("片段ID"),
                             "原因": "只有 %.2fs，短于参考音下限 %.1fs"
                                     % (got_sec, VOICE_BASE_MIN)})
            continue
        deal = _purify_voice_base(rec, dst, src, pool or {})
        gate = gates.check("克隆基准音", dst)
        if not gates.ok(gate):
            # 标注说有人声、实测近乎无声：拿去克隆只会克出噪音嗓，弃用换下一候选
            rejected.append({"片段": src.get("片段ID"),
                             "原因": gate.get("判据") or gate.get("原因", ""),
                             "门禁报告": gate})
            continue
        plan.update(deal)
        plan.update({"基准音文件": dst, "基准来源片段": src.get("片段ID"),
                     "基准来源说明": src.get("说明"),
                     "基准音响度dB": (gate.get("指标") or {}).get("平均响度dB"),
                     "基准音门禁": dict(gate, 使用=True)})
        try:
            plan["基准音URL"] = storage.upload(dst)
        except Exception as exc:  # noqa: BLE001
            # 上传失败只影响 AI 段能不能拿它当参考声音，本地克隆配音照常能用
            plan["上传失败"] = str(exc)[:150]
        break
    if rejected:
        plan["候选弃用"] = rejected
    if not plan["基准音文件"] and "用户视频-原声" in available:
        # 素材人声候选全军覆没：事实修正后重新按表判定（一般落到 AI 首段引导）
        available.remove("用户视频-原声")
        plan.update(rules.pick_voice(available))
        log(rec, "音色：素材人声候选全部弃用（%s），按修正事实重新判定"
            % "；".join(r["原因"] for r in rejected[-2:]))
    path = _p(tid, "audio", "voice_plan.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
    log(rec, "音色：%s（%s）%s" % (plan["策略"], plan.get("来源") or "无原声可用",
                                  ("基准音已就绪（%.1f dB），%s"
                                   % (plan.get("基准音响度dB") or 0, plan.get("基准来源说明")))
                                  if plan["基准音文件"] else "无基准原声"))
    return plan


DUB_MAX_TEMPO = 1.4          # 配音比画面长时最多加速多少倍（再快就明显失真）
DUB_MAX_STRETCH = 1.5        # 画面最多按计划时长放长多少倍（再长就压掉后面段落的节奏）
DUB_MIN_TEMPO = 0.85         # 配音比画面短时最多放慢多少倍（再慢就念成慢动作朗读）
DUB_GAP_MAX = 0.25           # 念白尾部允许留多长空档：超过它就先放慢念白去填
# 本机没有 GPU，VoxCPM2 单条合成就吃满 ~30 核（实测 4s 音频 131s）。并发跑三条会互相
# 拖到 600s 超时全灭，所以配音一次只跑一条；段内出片的并发交给网络端的生成任务去用。
_TTS_LOCK = threading.Lock()


def _join_lines(texts) -> str:
    """把几句台词接成一段念白：上一句自带句读就不再补逗号（否则会念出「呢，，我」）。"""
    out = ""
    for t in (x.strip() for x in texts if x and x.strip()):
        if out and out[-1] not in "，。！？；、…,.!?;":
            out += "，"
        out += t
    return out


def _dub_wav(rec: dict, block: dict, shots: dict, plan: dict, tag: str) -> dict:
    """先把这一块的台词用克隆音色念出来（先不贴回），返回 {"wav","dur","info"}。

    合成放在裁剪之前：先知道念白多长，才能决定画面要不要放长（见 _dub_scale）——
    分镜时长照的是参考片语速，TTS 念得慢，按计划时长裁完再贴必然砍掉半句话。
    台词按分镜取（去掉「角色：」前缀），否则 TTS 会把「女主：」也念出来。
    """
    text = _join_lines(_shot_text(shots.get(n)) for n in block["镜头"])
    base = plan.get("基准音文件") or ""
    if not text:
        return {"info": {"配音": "跳过：这一片没有台词"}}
    if not base or not tts_clone.available():
        return {"info": {"配音": "跳过：没有可克隆的基准原声"
                                 if not base else "跳过：TTS 后端未就绪"}}
    wav = _p(rec["task_id"], "audio", "dub_seg%s.wav" % tag)
    with _TTS_LOCK:                       # 一次只跑一条，见 _TTS_LOCK 的说明
        got = tts_clone.clone(base, "", text, wav)
    if not got.get("ok"):
        return {"info": {"配音": "失败：%s" % str(got.get("error"))[:200]}}
    dur = float(got.get("duration") or 0)
    return {"wav": wav, "dur": dur,
            "info": {"配音": "克隆原声配音", "配音文件": wav, "配音时长秒": round(dur, 2)}}


def _dub_scale(dub: float, planned: float) -> float:
    """念白塞不进计划时长时，画面要放长多少倍（先靠加速念白，再靠放长画面，各有上限）。"""
    if dub <= 0 or planned <= 0 or dub <= planned * DUB_MAX_TEMPO + 0.15:
        return 1.0
    return min(DUB_MAX_STRETCH, dub / DUB_MAX_TEMPO / planned)


def _paste_dub(rec: dict, piece: dict, dub: dict) -> dict:
    """把念白贴回这一片：长了先加速（上限 DUB_MAX_TEMPO）再截断，短了先放慢再补静音。

    短的那一头以前是直接 apad 补静音，成片里就是「画面在动、口播卡住」——9399 实测
    段1 念白 5.12s / 画面 5.62s、段2 13.12s / 13.86s，两处各留出 0.8~0.9s 空档，
    BGM 只有 -32dB 撑不住。所以先把念白放慢到画面长度（放慢 15% 以内听不出来），
    放慢到顶还差的那点才补静音。
    """
    info = dict(dub.get("info") or {})
    wav, need = dub.get("wav") or "", float(dub.get("dur") or 0)
    if not wav:
        return info
    vid = produce_video._duration(piece["file"])
    tempo = 1.0
    if vid > 0.2 and need > vid + 0.15:
        tempo = min(DUB_MAX_TEMPO, need / vid)
        info["配音变速"] = round(tempo, 3)
        if need / tempo > vid + 0.3:
            info["配音截断秒"] = round(need / tempo - vid, 2)
    elif vid > 0.2 and need > 0.2 and need < vid - DUB_GAP_MAX:
        tempo = max(DUB_MIN_TEMPO, need / vid)
        info["配音变速"] = round(tempo, 3)
        left = vid - need / tempo
        if left > DUB_GAP_MAX:
            info["尾部补静音秒"] = round(left, 2)
    # 音轨长度必须自己 atrim 到画面长度：apad + -shortest 会让 ffmpeg 挂死
    # （apad 是无限流，配 -c:v copy 时 -shortest 收不住，实测跑 5 分钟不结束）
    chain = "[1:a]%sapad,atrim=0:%.3f,asetpts=N/SR/TB[a]" % (
        ("atempo=%.4f," % tempo) if abs(tempo - 1.0) > 0.001 else "", max(0.2, vid))
    tmp = os.path.splitext(piece["file"])[0] + "_dub.mp4"
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                             "-i", piece["file"], "-i", wav, "-filter_complex", chain,
                             "-map", "0:v:0", "-map", "[a]", "-c:v", "copy"]
                            + produce_video.aac_args("160k") + [tmp])
    if ret.returncode != 0 or not os.path.isfile(tmp):
        info["配音"] = "贴回失败：%s" % ret.stderr[-200:]
        return info
    os.replace(tmp, piece["file"])
    info["时长秒"] = round(produce_video._duration(piece["file"]), 2)
    return info


SILENT_DB = gates.SILENT_DB  # 静音线唯一来源在 gates.py（实测门禁范式），此处只留别名


def _mean_db(path: str) -> float:
    """量一段媒体的平均音量（实现在 gates.py：实测门禁范式的指标原语）。"""
    return gates.mean_db(path)


def _fill_silent_dub(rec: dict, piece: dict, block: dict, shots: dict,
                     plan: dict, tag: str, label: str) -> None:
    """补片有台词却没声，就用克隆音色把这句话配上。

    seedance 时不时返回静音音轨（或干脆没有音轨），这样成片的口播就会凭空断掉一句；
    音色和别的片略有差别可以接受，少一句话不行。所以逐片检查、缺声就补。
    """
    if not any(_shot_text(shots.get(n)) for n in block["镜头"]):
        return                          # 这一片本来就没台词，静音是分镜设计
    # 走 gates.check 而不是内联比阈值：这样这条静音判定会产出标准门禁报告，
    # review_case 的 gates.violations() 对账才看得见它（gates.py 第 5 条范式）。
    gate = gates.check("有声内容", piece["file"])
    if gates.ok(gate):
        return
    db = (gate.get("指标") or {}).get("平均响度dB")
    dub = _dub_wav(rec, block, shots, plan, tag)
    piece.update(_paste_dub(rec, piece, dub))
    piece["补配音"] = "补片%s，视为没有人声" % (gate.get("判据") or gate.get("原因") or "")
    piece["有声内容门禁"] = dict(gate, 使用=True)
    log(rec, "  %s 补片没有人声（%s），用克隆音色补上：%s"
        % (label, "%.0f dB" % db if isinstance(db, (int, float)) else "判不出",
           piece.get("配音") or "未补上"))
