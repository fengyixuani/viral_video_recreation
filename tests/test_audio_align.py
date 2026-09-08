"""配音贴回与段间响度对齐的实测自检：python3 tests/test_audio_align.py

不调任何模型，只用 ffmpeg 造合成素材（正弦音 + testsrc 画面）跑三件事：
1. 段内按「需要配音」拆块——保留原声的镜头不能和要重配的镜头编进同一块；
2. 念白比画面短时先放慢填满，不再在尾部留出听得见的空档；
3. 进 concat 的每一片响度对齐到同一档，静音片不被抬起来。
"""
# 从仓库根目录跑：python3 tests/test_audio_align.py（下面两行让 import 找得到仓库模块）
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import gates
import produce_video
import segment_build
import voice_dub
from media import _ffmpeg

TMP = "/tmp/viralforge_audio_align"


def _mk_video(dst: str, sec: float, db: float = None) -> str:
    """造一条 sec 秒的测试视频：db 给了就带一路该音量的正弦音，否则整条静音。

    db 是下发给 volume 滤镜的值，不等于实测平均响度——本机 ffmpeg 的 sine 满幅实测
    约 -24.1 dB，两者线性相差 24.1 dB（volume=+4dB → 实测 -20.1dB）。测试要的是
    「实测响度落在某个档」，所以调用方按这个偏移换算，别直接把 db 当实测值断言。
    """
    audio = ("sine=frequency=220:duration=%.2f,volume=%.1fdB" % (sec, db) if db is not None
             else "anullsrc=channel_layout=stereo:sample_rate=44100")
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                              "-f", "lavfi", "-i", "testsrc=size=360x640:rate=30:duration=%.2f" % sec,
                              "-f", "lavfi", "-i", audio, "-t", "%.2f" % sec,
                              "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
                             + produce_video.aac_args() + [dst])
    assert ret.returncode == 0 and os.path.isfile(dst), "造测试视频失败：%s" % ret.stderr[-200:]
    return dst


def _mk_wav(dst: str, sec: float, db: float = 0.0) -> str:
    """造一条 sec 秒的测试念白（正弦音）。db 同 _mk_video：是 volume 值，不是实测响度。"""
    ret = produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                              "-f", "lavfi",
                              "-i", "sine=frequency=330:duration=%.2f,volume=%.1fdB" % (sec, db),
                              "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", dst])
    assert ret.returncode == 0 and os.path.isfile(dst), "造测试念白失败：%s" % ret.stderr[-200:]
    return dst


def _tail_db(path: str, tail: float = 0.6) -> float:
    """最后 tail 秒的平均响度：尾部空档就是靠这个量出来的。"""
    dur = produce_video._duration(path)
    cut = os.path.join(TMP, "tail.wav")
    produce_video._run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", "%.2f" % max(0.0, dur - tail), "-i", path, "-vn",
                        "-c:a", "pcm_s16le", cut])
    return gates.mean_db(cut)


def t_blocks():
    """镜1 要重配、镜2 保留原声 → 必须拆成两块，否则块级配音会盖掉镜2 的原声。"""
    rows = [{"序号": 1, "策略": "直接裁剪", "源文件": "/x.mp4", "可用秒": 3,
             "切片动作": "静音后使用", "需要配音": True},
            {"序号": 2, "策略": "直接裁剪", "源文件": "/x.mp4", "可用秒": 3,
             "切片动作": "原声直接使用", "需要配音": False}]
    blocks = segment_build._seg_blocks({"镜头序号": [1, 2]}, rows)
    print("[blocks] %d 块 -> %s" % (len(blocks),
                                    [(b["镜头"], b["需要配音"]) for b in blocks]))
    assert len(blocks) == 2, "保留原声的镜头被编进了要配音的块"
    assert blocks[0]["需要配音"] is True and blocks[1]["需要配音"] is False
    # 同一动作的连续镜头仍要并块（不能退化成一镜一块，那会多花生成额度）
    same = segment_build._seg_blocks(
        {"镜头序号": [1, 2]},
        [dict(rows[0], 序号=n) for n in (1, 2)])
    assert len(same) == 1, "同为重配音的连续镜头不该拆块"
    print("[blocks] ok")


def t_paste_dub_short():
    """念白 3.7s / 画面 4.0s（9399 实测就是这个量级）：放慢后填满，尾部不能是静音。"""
    vid, wav = _mk_video(os.path.join(TMP, "short.mp4"), 4.0), _mk_wav(os.path.join(TMP, "short.wav"), 3.7)
    piece = {"file": vid}
    info = voice_dub._paste_dub({}, piece, {"wav": wav, "dur": 3.7, "info": {"配音": "克隆原声配音"}})
    dur, tail = produce_video._duration(vid), _tail_db(vid)
    print("[short] 变速=%s 尾部补静音=%s 成片 %.2fs 尾部 %.1f dB"
          % (info.get("配音变速"), info.get("尾部补静音秒"), dur, tail))
    assert abs((info.get("配音变速") or 0) - 3.7 / 4.0) < 0.01, \
        "没有按画面长度放慢念白：%s" % info.get("配音变速")
    assert not info.get("尾部补静音秒"), "放慢后还留了空档：%s" % info.get("尾部补静音秒")
    assert tail > gates.SILENT_DB, "尾部还是空档（%.1f dB）" % tail
    print("[short] ok")


def t_paste_dub_too_short():
    """念白只有画面的 3/4：放慢到 DUB_MIN_TEMPO 就得停手，剩下的空档必须留痕。"""
    vid, wav = _mk_video(os.path.join(TMP, "tiny.mp4"), 4.0), _mk_wav(os.path.join(TMP, "tiny.wav"), 3.0)
    info = voice_dub._paste_dub({}, {"file": vid}, {"wav": wav, "dur": 3.0, "info": {}})
    left = info.get("尾部补静音秒") or 0
    print("[too_short] 变速=%s 尾部补静音=%.2fs（不放慢会留 1.00s）"
          % (info.get("配音变速"), left))
    assert info.get("配音变速") == voice_dub.DUB_MIN_TEMPO, "放慢没有卡在下限"
    assert 0 < left < 1.0 - 0.3, "空档没被压下来，也没留痕：%s" % info
    print("[too_short] ok")


def t_paste_dub_long():
    """念白 6.0s / 画面 4.0s：仍走加速分支（回归，别被短分支抢了）。"""
    vid, wav = _mk_video(os.path.join(TMP, "long.mp4"), 4.0), _mk_wav(os.path.join(TMP, "long.wav"), 6.0)
    info = voice_dub._paste_dub({}, {"file": vid}, {"wav": wav, "dur": 6.0, "info": {}})
    print("[long] 变速=%s 截断=%s" % (info.get("配音变速"), info.get("配音截断秒")))
    assert (info.get("配音变速") or 0) > 1.0, "念白比画面长时没有加速"
    print("[long] ok")


def t_normalize_loudness():
    """两片实测响度差 9 dB（≈ -25 / -16，跨过目标档两侧），归一化后都要落到目标档；静音片保持静音。"""
    quiet = _mk_video(os.path.join(TMP, "quiet.mp4"), 2.0, db=-1.0)    # 实测约 -25.1 dB
    loud = _mk_video(os.path.join(TMP, "loud.mp4"), 2.0, db=8.0)       # 实测约 -16.1 dB
    silent = _mk_video(os.path.join(TMP, "silent.mp4"), 2.0)
    out = [os.path.join(TMP, "n_%s.mp4" % n) for n in ("quiet", "loud", "silent")]
    for src, dst in zip((quiet, loud, silent), out):
        assert produce_video._normalize(src, dst), "归一化失败：%s" % src
    before = [gates.mean_db(p) for p in (quiet, loud, silent)]
    after = [gates.mean_db(p) for p in out]
    print("[loudness] 归一前 %s -> 归一后 %s（目标 %.1f）"
          % (["%.1f" % d for d in before], ["%.1f" % d for d in after],
             produce_video.TARGET_MEAN_DB))
    assert abs(after[0] - after[1]) < 1.5, "两片响度没对齐：%.1f vs %.1f" % (after[0], after[1])
    for got in after[:2]:
        assert abs(got - produce_video.TARGET_MEAN_DB) < 1.5, "没落到目标档：%.1f" % got
    assert after[2] <= gates.SILENT_DB, "静音片被抬起来了：%.1f dB" % after[2]
    print("[loudness] ok")


ALL = {"blocks": t_blocks, "short": t_paste_dub_short, "too_short": t_paste_dub_too_short,
       "long": t_paste_dub_long, "loudness": t_normalize_loudness}


def main(argv):
    os.makedirs(TMP, exist_ok=True)
    names = argv or list(ALL)
    for name in names:
        ALL[name]()
    print("全部通过：%s" % "、".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))
