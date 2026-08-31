"""纯重剪链路：把爆款的剪辑还原到用户素材上，零模型调用、零 token 成本。

和主链路（pipeline.py 九步）的关系：这条链路只留「拆爆款结构 → 找素材 → 裁剪拼接」，
砍掉 product（商品事实卡）、script（剧本改写）、AIGC 补片。成立前提是**爆款本身就是从这份
用户素材里剪出来的**（短剧投放素材的常见形态）——此时「找素材」根本不需要语义理解：两边画面
是同一批帧，用感知哈希做帧级检索，比让 VLM 各自描述再配对又快又准。

为什么不调模型：
- 重剪要的事实只有两个：爆款的镜头切点、每个镜头在素材里的位置。都是像素级事实，
  VLM 给的是语义描述，反而要靠时间码对齐，绕远且贵。
- analyze_materials 走 VLM 要把整条素材 base64 内联上传，59 分钟原片实测跑 4.6 分钟未返回。
- 显式失败：指纹匹配不上的镜头记「未命中」写进 EDL，不拿相近画面糊过去。

链路：
0) box     几何归一化：探出内容区，裁掉信箱黑边与标题/落款带（横屏原片塞进竖画布的必需步骤）
1) shots   爆款镜头切分：相邻帧突变 = 硬切；定位失败的镜头再内部找叠化点补一刀
2) index   素材指纹索引：同样方式解码用户素材，建 (时间, hash) 时间轴
3a) asr    先跑台词：爆款与素材各做一遍词级 ASR，n-gram 投票连成单调锚点表，
           把每一镜的搜索范围从整条原片缩到锚点附近几十秒（台词不随改分辨率/加字幕/贴纸/变速而变）
3b) hash   再滑窗：镜头整段在锚点窄窗里序列对齐定精确帧（含变速搜索、避开转场帧）；
           窄窗里只勉强对上时再搜一遍全片，但全片解要明显更好才敢推翻锚点的时序约束
3c) vlm    哈希也对不上才上：抽 3 帧问多模态大模型，只校验候选、不让它满片检索
3d) aigc   三层全失败、且显式开了 --aigc 才上：按爆款画面生成一段补上（默认关）
4) cut     按定位点从素材裁剪，变速镜按倍率取料再压回原时长，统一到成片画布
5) concat  流复制拼接成 final.mp4

跑法：python3 recut.py <爆款片> <用户素材...> [--outdir DIR] [--mute]
产物：<outdir>/final.mp4、edl.json（逐镜源位置与命中距离）、report.md、pieces/
"""
import argparse
import json
import os
import re
import subprocess
import time

import numpy as np

import produce_video as pv  # pyright: ignore[reportImplicitRelativeImport]

# ---------------- 参数：全是像素级阈值，集中在这里 ----------------
HASH_W, HASH_H = 9, 8         # dHash 网格：9 列相邻两两比较，出 8×8=64 bit
BOTTOM_CROP = 0.22            # 内容区里裁掉底部的比例：短剧素材底部烧字幕，两版字幕不同会污染指纹
SIDE_CROP = 0.08              # 左右各裁掉的比例：投放素材常在侧边压竖排引流水印
TOP_CROP = 0.06               # 顶部裁掉的比例：角标 / 台标 / 贴纸
EDGE_SKIP = 0.12              # 每镜取样避开首尾这个比例：软转场（叠化/淡入）的混合帧都在镜头两端
# 变速候选，1.0 放最前面：分数打平时优先判成没变速（严格小于才替换最优解）
SPEEDS = (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.25, 0.75, 1.5, 2.0)
REF_FPS = 10.0                # 爆款解码帧率：要定位切点，采密一点
INDEX_FPS = 4.0               # 素材解码帧率：只用来定位，4fps 够用且省一半解码时间
CUT_DIST = 20                 # 相邻帧汉明距离 > 此值判为硬切
SOFT_LAG = 0.4                # 软转场探测的跨帧窗口（秒）：叠化逐帧变化很小，要跨几帧才看得出来
SOFT_DIST = 20                # 跨窗汉明距离 > 此值、且窗内没有硬切，判为软转场
SPEED_MARGIN = 1.5            # 非 1.0 变速必须比 1.0 好这么多分才采纳，避免静止镜头在 1.0/1.05 间抖动
MIN_SHOT_SEC = 0.4            # 短于此的镜头并进前一镜（转场闪帧不算一个镜头）
TAPS = (3, 12)                # 每镜取样帧数的下限/上限（按镜头时长在区间内取）
MEAN_MATCH_DIST = 12.0        # 整镜序列对齐后的平均汉明距离 > 此值判未命中
ASR_MIN_CHARS = 4             # 台词少于此字数不走 ASR：短于 4-gram 对不上
ASR_PAD = 8.0                 # 邻镜源区间外扩这么多秒再转写：给变速/剪辑误差留余量
ASR_MAX_WIN = 90.0            # 素材短窗上限：CPU 上约 1x 实时，再长就退化成整段转写
ASR_CHUNK = 30.0              # 整片转写的分块长度：Qwen3-ASR max_new_tokens=512，整条 34 分钟一次喂进去会被截断
ANCHOR_NGRAM = 5              # 台词锚点用几连字投票：4 连字在短剧里重复太多
ANCHOR_PAD = 6.0              # 锚点给哈希滑窗留的基础半径（秒）
ANCHOR_MIN_VOTES = 2          # 一个锚点至少要几处 n-gram 同时投出来
ANCHOR_DRIFT = 0.25           # 每离最近锚点 1s，窗再放宽这么多秒（锚点间插值的漂移速率）
ANCHOR_TRUST = 6.0            # 锚点窗内平均距离低于此值就直接采信，不再多搜一遍全片
ANCHOR_MARGIN = 4.0           # 全片解要比锚点解好这么多分才敢推翻锚点（分数接近时信锚点的时序约束）
VLM_MAX_SHOTS = 8             # 一条爆款最多问这么多次 VLM，挡住异常未命中把成本打爆
AIGC_MAX_SHOTS = 6            # 最多生成这么多镜：AIGC 是最后一手，多了就不是「重剪」了
BOX_SAMPLES = 12              # 探内容区取几帧
BOX_GRID = 64                 # 探内容区的取样网格
BAR_LEVEL = 0.18              # 行/列平均亮度低于峰值这个比例的，算黑边或标题/落款带

_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)


# ---------------- 指纹 ----------------
def _hamming(hashes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """一批 hash 与一个 hash 的汉明距离。hash 都是 (n, 8) uint8。"""
    return _POP[np.bitwise_xor(hashes, query)].sum(axis=1)


def _box_text(box: tuple) -> str:
    return "x%.0f%% y%.0f%% w%.0f%% h%.0f%%" % tuple(v * 100 for v in box)


def _video_wh(path: str) -> tuple:
    """从 ffmpeg 的 Video: 行读宽高，读不到返回 (1, 1)（后面按比例算，1:1 等于不校正）。"""
    for line in pv._run([pv._ffmpeg(), "-hide_banner", "-i", path]).stderr.splitlines():
        if "Video:" not in line:
            continue
        for tok in line.replace(",", " ").split():
            if "x" in tok and tok[0].isdigit():
                a, b = tok.split("x", 1)
                if a.isdigit() and b.isdigit() and int(a) > 0 and int(b) > 0:
                    return int(a), int(b)
    return 1, 1


# ---------------- 步骤 0：几何归一化（内容区探测）----------------
def content_box(path: str) -> tuple:
    """探出画面里真正有内容的区域，返回 (x, y, w, h) 四个比例。

    为什么必须有这一步：短剧投放素材常把横屏原片**信箱化**塞进 9:16 竖画布，上下补黑边，
    再压一条标题带和一条「内容由AI生成」落款带。这种爆款和横屏原片逐帧比指纹永远匹配不上——
    竖版画面里大半是黑边，指纹被黑边主导（实测 r2_001 / r4_003 两条竖版爆款命中 0 镜，
    而同一条原片剪出的横版 r2_022 命中 94%）。把两边都先裁到内容区，几何就对齐了。

    做法：均匀取几帧求平均灰度图，逐行/逐列求平均亮度，取**最长的一段连续亮区**当内容区。
    只卡阈值不够：标题带是彩色大字，整行平均亮度能到峰值的 60%，比不上内容区但远高于黑边
    （实测 r2_001 行剖面 标题行 46 / 内容行 56-75 / 黑边 1 / 落款行 13）。而标题带只占一两行、
    与内容区之间隔着纯黑行，按最长连续段取就能把它和落款带一起排除掉。
    """
    dur = max(1.0, pv._duration(path))
    vf = ("fps=%.6f,scale=%d:%d:flags=area,format=gray"
          % (BOX_SAMPLES / dur, BOX_GRID, BOX_GRID))
    proc = subprocess.run([pv._ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
                           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                          capture_output=True, check=False)
    cell = BOX_GRID * BOX_GRID
    count = len(proc.stdout) // cell
    if not count:
        return (0.0, 0.0, 1.0, 1.0)
    mean = np.frombuffer(proc.stdout[:count * cell], dtype=np.uint8)
    mean = mean.reshape(count, BOX_GRID, BOX_GRID).astype(np.float32).mean(axis=0)

    def span(profile: np.ndarray) -> tuple:
        """取最长的一段连续亮区（起, 止）。"""
        peak = float(profile.max())
        if peak <= 0.0:
            return 0, len(profile)
        runs, start = [], None
        for i, lit in enumerate(profile >= peak * BAR_LEVEL):
            if lit and start is None:
                start = i
            elif not lit and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(profile)))
        return max(runs, key=lambda r: r[1] - r[0]) if runs else (0, len(profile))

    top, bottom = span(mean.mean(axis=1))
    left, right = span(mean.mean(axis=0))
    grid = float(BOX_GRID)
    return (left / grid, top / grid, (right - left) / grid, (bottom - top) / grid)


def _margin_crop(box: tuple) -> tuple:
    """内容区里再切掉四边字幕/水印/角标。"""
    return (box[0] + box[2] * SIDE_CROP, box[1] + box[3] * TOP_CROP,
            box[2] * (1.0 - 2.0 * SIDE_CROP), box[3] * (1.0 - TOP_CROP - BOTTOM_CROP))


def _fit_ar(crop: tuple, vw: int, vh: int, target_ar: float) -> tuple:
    """把 crop 的像素长宽比收到 target_ar（取更窄的那块中心）。

    投放素材常把 16:9 原片左右裁成 4:3/1.3 塞进竖画布。两边内容区长宽比不同时，
    各自缩到 9×8 等于拿不同画面块去比。共同长宽比取「这一组里最窄的」，
    16:9 对 16:9 信箱（r4）不会再裁；16:9 对 1.3 中心裁（r2_001）只裁原片两侧。
    """
    pix_w, pix_h = crop[2] * vw, crop[3] * vh
    if pix_w <= 0 or pix_h <= 0 or target_ar <= 0:
        return crop
    cur = pix_w / pix_h
    if cur > target_ar * 1.05:
        nw = (target_ar * pix_h) / vw
        return (crop[0] + (crop[2] - nw) / 2.0, crop[1], nw, crop[3])
    if cur * 1.05 < target_ar:
        nh = (pix_w / target_ar) / vh
        return (crop[0], crop[1] + (crop[3] - nh) / 2.0, crop[2], nh)
    return crop


def decode_hashes(path: str, fps: float, cache_dir: str = "",
                  target_ar: float = 0.0) -> tuple:
    """解码成 dHash 时间轴，返回 (times, hashes, box)。

    先按 content_box 裁到内容区做几何归一化，再在内容区内部裁掉底部字幕带，
    最后让 ffmpeg 缩到 9×8 灰度（area 滤波 = 盒式平均，等于先降噪再取样）。
    一条 59 分钟视频在 4fps 下也只有 ~1MB 原始数据，全放内存里算。
    指纹只由 (文件内容, fps, 几何参数) 决定，所以按这些做 key 落盘缓存：同一条素材被多个
    爆款复用时（一条原片剪出好几条投放素材）不用重复解码，整条链路最贵的一步只付一次。
    """
    box = content_box(path)
    crop = _margin_crop(box)
    vw, vh = _video_wh(path)
    if target_ar > 0:
        crop = _fit_ar(crop, vw, vh, target_ar)
    key = ""
    if cache_dir:
        stat = os.stat(path)
        key = os.path.join(cache_dir, "%s-%d-%d-%.2f-%d-%d-%s.npz"
                           % (os.path.basename(path), stat.st_size, int(stat.st_mtime),
                              fps, HASH_W, HASH_H,
                              "-".join("%.4f" % v for v in crop)))
        if os.path.isfile(key):
            got = np.load(key)
            return got["times"], got["hashes"], box
    vf = ("crop=iw*%.6f:ih*%.6f:iw*%.6f:ih*%.6f,fps=%.4f,scale=%d:%d:flags=area,format=gray"
          % (crop[2], crop[3], crop[0], crop[1], fps, HASH_W, HASH_H))
    cmd = [pv._ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    size = HASH_W * HASH_H
    count = len(proc.stdout) // size
    if proc.returncode != 0 and not count:
        raise RuntimeError("解码失败 %s：%s"
                           % (os.path.basename(path),
                              proc.stderr[-300:].decode("utf-8", "replace")))
    if not count:
        raise RuntimeError("解码没拿到任何帧：%s" % path)
    grid = np.frombuffer(proc.stdout[:count * size], dtype=np.uint8)
    grid = grid.reshape(count, HASH_H, HASH_W).astype(np.int16)
    bits = (grid[:, :, 1:] > grid[:, :, :-1]).reshape(count, (HASH_W - 1) * HASH_H)
    times, hashes = np.arange(count, dtype=np.float64) / fps, np.packbits(bits, axis=1)
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(key, times=times, hashes=hashes)
    return times, hashes, box


# ---------------- 步骤 1：爆款镜头切分 ----------------
def split_shots(times: np.ndarray, hashes: np.ndarray, fps: float) -> list:
    """相邻帧指纹突变处切硬切；过短的并进前一镜。返回 [(起, 止)] 秒。

    只认硬切。叠化/淡入淡出的逐帧变化过小，CUT_DIST 抓不到——那一类留给
    `soft_cut_at`：定位失败才去镜头内部找软转场，避免把运动镜头切碎
    （r1_030 上无差别软切会多出 100+ 刀，命中率反而掉）。
    """
    tail = 1.0 / fps
    if len(hashes) < 2:
        return [(0.0, float(times[-1]) + tail if len(times) else 0.0)]
    jump = _hamming(hashes[1:], hashes[:-1]) > CUT_DIST
    edges = [0] + [i + 1 for i, hit in enumerate(jump) if hit] + [len(hashes)]
    shots: "list[tuple[float, float]]" = []
    for a, b in zip(edges[:-1], edges[1:]):
        start, end = float(times[a]), float(times[b - 1]) + tail
        if shots and end - start < MIN_SHOT_SEC:
            shots[-1] = (shots[-1][0], end)
        else:
            shots.append((start, end))
    return shots


def soft_cut_at(shot: tuple, times: np.ndarray, hashes: np.ndarray, fps: float):
    """定位失败的镜头里找一个叠化点。找不到返回 None。

    叠化的特征：相邻帧距离始终低于硬切阈值，但跨 SOFT_LAG 秒后画面已经换了。
    只在「滑窗对齐已经失败」时调用，所以误切的代价只是把一段本就未命中的
    区间再剖一刀，不会动已经命中的镜头。
    """
    start, end = shot
    if end - start < 2.5:                    # 两段源画面被 0.5s 叠化并在一起，最短也有好几秒
        return None
    i0 = int(np.searchsorted(times, start))
    i1 = min(len(hashes), int(np.searchsorted(times, end)))
    lag = max(1, int(round(SOFT_LAG * fps)))
    if i1 - i0 <= lag + 2:
        return None
    far = _hamming(hashes[i0 + lag:i1], hashes[i0:i1 - lag])
    pad = max(1, int(round(len(far) * EDGE_SKIP)))
    if len(far) <= 2 * pad:
        return None
    inner = far[pad:-pad]
    peak_rel = int(np.argmax(inner))
    peak = float(inner[peak_rel])
    if peak < SOFT_DIST or peak < float(np.median(inner)) + 8.0:
        return None
    peak_i = pad + peak_rel
    lo, hi = peak_i, peak_i
    while lo > 0 and far[lo - 1] >= peak * 0.7:
        lo -= 1
    while hi + 1 < len(far) and far[hi + 1] >= peak * 0.7:
        hi += 1
    if (hi - lo + 1) > lag * 3:          # 长高原 = 持续运动，不是一次叠化
        return None
    cut = float(times[min(len(times) - 1, i0 + peak_i + lag // 2)])
    if cut - start < MIN_SHOT_SEC or end - cut < MIN_SHOT_SEC:
        return None
    return cut


# ---------------- 步骤 3：逐镜定位回素材 ----------------
def _align(query: np.ndarray, offsets: np.ndarray, idx_hashes: np.ndarray) -> tuple:
    """把一组取样帧按给定的源侧帧偏移在素材索引上滑窗，返回 (最佳帧位置, 平均汉明距离)。
    位置 -1 表示素材比这段还短，无解。"""
    room = len(idx_hashes) - int(offsets[-1])
    if room <= 0:
        return -1, 64.0
    total = np.zeros(room, dtype=np.int32)
    for i, off in enumerate(offsets):
        total += _hamming(idx_hashes[int(off):int(off) + room], query[i])
    best = int(np.argmin(total))
    return best, float(total[best]) / len(offsets)


def locate(shot: tuple, ref_times: np.ndarray, ref_hashes: np.ndarray,
           idx_times: np.ndarray, idx_hashes: np.ndarray, window=None) -> dict:
    """把一个爆款镜头整段对齐到素材时间轴，返回最佳起点、平均汉明距离和推断出的变速倍率。

    做的是**序列对齐**而不是单帧最近邻：镜头内取 k 个取样帧，按同样的相对间隔在素材
    时间轴上滑窗，取「各取样帧距离之和」最小的位置。
    早先的实现是每个取样帧各自找最近邻、再要求反推起点一致，实测大量真命中被误判成未命中：
    镜头内画面几乎静止时，一帧在素材里能匹配上整段静止区间里的任意一帧，反推起点自然散开
    （距离只有 1-5 却离散 5s）。滑窗对齐把「镜内相对时序」当成约束，静止镜头也只有一个解。

    两个针对真实投放素材的处理：
    - 转场：取样避开镜头两端 EDGE_SKIP，叠化/淡入淡出的混合帧都在那儿，拿它们当查询帧
      等于拿噪声去比。
    - 变速：滑窗的帧间隔乘上候选倍率再比。倍率定义为 源时长/爆款时长，爆款放快了就 >1。
    """
    start, end = shot
    dur = max(0.0, end - start)
    taps = int(min(TAPS[1], max(TAPS[0], round(dur * INDEX_FPS))))
    pad = dur * EDGE_SKIP
    lo, hi = start + pad, end - pad - 1.0 / REF_FPS
    if hi <= lo:                                    # 镜头太短，内缩后没剩余，退回全段取样
        lo, hi = start, max(start, end - 1.0 / REF_FPS)
    rel = np.linspace(lo, hi, taps)
    query = np.stack([ref_hashes[int(np.argmin(np.abs(ref_times - t)))] for t in rel])
    lead, step = rel[0] - start, rel - rel[0]       # lead 用于把匹配位置回算成镜头起点

    # 台词锚点给出的窄窗：只在这一段索引里滑。既省算力，又挡掉「别处有张几乎一样的脸」
    # 这类远距离误命中——短剧同一场景反复出现，全片搜时相似画面是主要误差来源。
    # 窗的语义是「镜头起点可能落在 [lo, hi]」，所以尾部要再放出一整个镜头的长度（乘上最大
    # 变速倍率），否则 hi 会把查询序列的尾巴切掉：锚点只偏 1.5s，滑窗就找不到真解，
    # 只能退而报一个分数差得多的位置（r1_011 镜39 真解 1.8 分被压成 5.5 分就是这么来的）。
    if window is not None:
        span = float(step[-1]) * max(SPEEDS) if len(step) else 0.0
        a = int(np.searchsorted(idx_times, max(0.0, window[0])))
        b = int(np.searchsorted(idx_times, window[1] + span)) + 1
        if b - a >= taps + 2:          # 切片保留绝对时间，pack() 回算起点不用改
            idx_times, idx_hashes = idx_times[a:b], idx_hashes[a:b]
        else:
            window = None              # 窗太窄放不下取样帧，退回全片搜

    def pack(speed: float, pos: int, score: float) -> dict:
        return {"命中": bool(score <= MEAN_MATCH_DIST),
                "源起点": round(max(0.0, float(idx_times[pos]) - lead * speed), 2),
                "平均距离": round(score, 2), "取样帧": taps, "变速": speed,
                "定位": "hash+asr锚" if window is not None else "hash"}

    # 默认按 1.0 对齐。变速只在 1.0 已经对不上、且镜头够长时才搜：
    # 投放素材真变速时动作对不齐，1.0 距离会过线；慢镜头 / 静止镜在 1.0 下照样命中，
    # 此时 1.25x/1.5x 也能再抠出 1-2 分，那是假阳性（r1_030 连续镜头源起点按 1.0 衔得上，
    # 却有 29 镜被标成 1.5x/2.0x，成片节奏全乱）。假阴性的代价只是少压一段倍速。
    pos0, score0 = _align(query, np.round(step * INDEX_FPS).astype(int), idx_hashes)
    base = pack(1.0, pos0, score0) if pos0 >= 0 else None
    if base is None:
        return {"命中": False, "源起点": 0.0, "平均距离": 64.0, "取样帧": taps, "变速": 1.0,
                "定位": "hash"}
    if base["命中"] or dur < 1.5:
        return base
    best = base
    for speed in SPEEDS:
        if speed == 1.0:
            continue
        pos, score = _align(query, np.round(step * speed * INDEX_FPS).astype(int), idx_hashes)
        if pos < 0:
            continue
        if best is None or score < best["平均距离"]:
            best = pack(speed, pos, score)
    if best is not None and base is not None and best["变速"] != 1.0:
        if base["平均距离"] - best["平均距离"] < SPEED_MARGIN:
            best = base
    return best or {"命中": False, "源起点": 0.0, "平均距离": 64.0,
                    "取样帧": taps, "变速": 1.0, "定位": "hash"}


# ---------------- 步骤 3b：哈希未命中才上 ASR ----------------
_ASR_PUNCT = set("，,。.!！？?;；：:、…~—－· \"'“”‘’()（）【】[]{}《》<> ")


def _asr_ready() -> bool:
    """本机 Qwen3-ASR 环境和权重是否齐全。缺了就跳过，不挡哈希链路。"""
    try:
        from Agent_tools.caption_clone.core import asr_tokens as tok
    except Exception:
        return False
    return bool(tok.available() and tok.ASR_ENTRY and os.path.isfile(tok.ASR_ENTRY)
                and tok.QWEN3_ASR_MODEL and os.path.isdir(tok.QWEN3_ASR_MODEL))


def _asr_clean(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _ASR_PUNCT)


def _asr_transcribe(video: str, start: float, dur: float, cache_dir: str) -> list:
    """把 [start, start+dur] 抽成 16k 单声道，跑词级 ASR，返回 [{text,start,end}]（相对窗起点）。

    本机没 GPU：实测 8s 音频 CPU 推理 8.5s。所以只转写爆款整段和未命中镜附近的短窗，
    59 分钟原片绝不整段送进去。结果按 (文件, 起点, 时长) 落盘，同一条爆款重跑只付一次。
    """
    if dur < 0.4 or not _asr_ready():
        return []
    from Agent_tools.caption_clone.core import asr_tokens as tok
    start, dur = max(0.0, start), max(0.4, dur)
    work = os.path.join(cache_dir or "/tmp", "asr")
    os.makedirs(work, exist_ok=True)
    key = "%s-%.2f-%.2f" % (os.path.basename(video), start, dur)
    raw = os.path.join(work, key + ".json")
    if os.path.isfile(raw):
        try:
            payload = json.load(open(raw, encoding="utf-8"))
            return ((payload.get("results") or [{}])[0].get("time_stamps") or {}).get("items") or []
        except (OSError, ValueError, TypeError):
            pass
    wav = os.path.join(work, key + ".wav")
    ret = pv._run([pv._ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", video,
                   "-vn", "-ac", "1", "-ar", "16000", wav])
    if ret.returncode != 0 or not os.path.isfile(wav):
        return []
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = os.path.join(work, "numba")
    env.setdefault("QWEN3_ASR_DEVICE", "cpu")
    env.setdefault("QWEN3_ASR_DTYPE", "float32")
    os.makedirs(env["NUMBA_CACHE_DIR"], exist_ok=True)
    cmd = [tok.ASR_PYTHON, tok.ASR_ENTRY, wav, raw, "--timestamps"]
    if tok.QWEN3_ASR_MODEL:
        cmd += ["--model", tok.QWEN3_ASR_MODEL]
    if tok.QWEN3_FORCED_ALIGNER:
        cmd += ["--forced-aligner", tok.QWEN3_FORCED_ALIGNER]
    try:
        subprocess.run(cmd, env=env, check=True,
                       timeout=float(os.getenv("ASR_TIMEOUT", "180")),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as exc:
        print("  ASR 失败 %s：%s" % (key, str(exc)[:160]), flush=True)
        return []
    try:
        payload = json.load(open(raw, encoding="utf-8"))
        return ((payload.get("results") or [{}])[0].get("time_stamps") or {}).get("items") or []
    except (OSError, ValueError, TypeError):
        return []


def _asr_chars(items: list, origin: float = 0.0) -> list:
    """[{text,start_time,end_time}] -> [{ch, t}]，t 是落到整条视频上的绝对秒。"""
    out = []
    for it in items or []:
        tok = _asr_clean(str(it.get("text") or ""))
        if not tok:
            continue
        t0 = origin + float(it.get("start_time") or it.get("start") or 0.0)
        t1 = origin + float(it.get("end_time") or it.get("end") or t0)
        if len(tok) == 1:
            out.append({"ch": tok, "t": t0})
            continue
        step = (t1 - t0) / len(tok)
        for i, ch in enumerate(tok):
            out.append({"ch": ch, "t": t0 + i * step})
    return out


def _ngram_hits(query: str, hay: list, n: int = 4) -> list:
    """query 在 hay[{ch,t}] 上滑 4-gram，返回按命中次数排序的绝对时间列表。"""
    q = _asr_clean(query)
    if len(q) < n or len(hay) < n:
        return []
    pos = {}
    for i in range(len(hay) - n + 1):
        key = "".join(h["ch"] for h in hay[i:i + n])
        pos.setdefault(key, []).append(hay[i]["t"])
    votes = {}
    for i in range(len(q) - n + 1):
        for t in pos.get(q[i:i + n], []):
            bucket = round(t - i * 0.12, 1)   # 回推到 query 开头，0.12s/字是中文语速
            votes[bucket] = votes.get(bucket, 0) + 1
    return sorted(votes, key=lambda t: (-votes[t], t))


def locate_by_asr(shot: tuple, query: str, mat: dict, win: tuple) -> dict:
    """在素材 [win0, win1] 里用台词 n-gram 定位一镜。命中时返回与 locate() 同形的 dict。"""
    start, end = shot
    dur = max(0.0, end - start)
    lo, hi = max(0.0, win[0]), max(win[0], win[1])
    if hi - lo > ASR_MAX_WIN:                 # 邻镜对不上、窗口被撑成整段时放弃，交给 VLM
        return {}
    items = _asr_transcribe(mat["file"], lo, hi - lo, mat.get("asr_cache") or "")
    hits = _ngram_hits(query, _asr_chars(items, lo))
    if not hits:
        return {}
    origin = max(0.0, hits[0])
    return {"命中": True, "源起点": round(origin, 2), "平均距离": 0.0,
            "取样帧": 0, "变速": 1.0, "定位": "asr",
            "源文件": os.path.basename(mat["file"]),
            "_src": mat["file"], "_silent": mat["silent"]}


def asr_stream(video: str, dur: float, cache_dir: str, tag: str = "") -> list:
    """整片词级 ASR，返回 [{ch, t}]（t 是绝对秒）。分块跑，每块单独落盘缓存。

    必须分块：run_qwen3_asr_test 的 max_new_tokens=512，一条 34 分钟的原片一次喂进去
    会在几百字处截断，后面全丢。按 ASR_CHUNK 切，块内时间戳加上块起点就是绝对时间。
    本机没 GPU，约 1x 实时——所以这一步只对**素材**做一次并落盘，一条原片剪出的
    所有爆款共用同一份转写。
    """
    chars, t, clock = [], 0.0, time.time()
    while t < dur:
        span = min(ASR_CHUNK, dur - t)
        chars.extend(_asr_chars(_asr_transcribe(video, t, span, cache_dir), t))
        t += span
        if tag and int(t) % 300 < ASR_CHUNK:
            print("  %s ASR %.0f/%.0fs → %d 字（%.0fs）"
                  % (tag, t, dur, len(chars), time.time() - clock), flush=True)
    return chars


def anchor_map(ref_chars: list, src_chars: list) -> list:
    """把爆款台词流对到素材台词流上，返回 [(爆款秒, 素材秒)] 的单调锚点表。

    做法：素材侧建 n-gram 倒排，爆款侧每个 n-gram 去投票，落到 0.5s 的桶里；
    票数够的桶留下，再按「爆款时间递增 → 素材时间也得递增」筛成单调序列。
    单调性这一步不能省：短剧里「你干什么」这种台词满片都是，单个 n-gram 会投到
    十几个位置，只有连成一条递增链的那些才是真正的剪辑对应关系。

    这份锚点表不负责精确定位，只负责把每个镜头的搜索范围从整条原片缩到 ±ANCHOR_PAD 秒，
    后面的哈希滑窗在窄窗里跑既快又不容易撞上相似画面。台词覆盖不到的静默镜头没有锚点，
    照旧全片搜。
    """
    if len(ref_chars) < ANCHOR_NGRAM or len(src_chars) < ANCHOR_NGRAM:
        return []
    n = ANCHOR_NGRAM
    index = {}
    for i in range(len(src_chars) - n + 1):
        key = "".join(c["ch"] for c in src_chars[i:i + n])
        index.setdefault(key, []).append(src_chars[i]["t"])
    votes = {}
    for i in range(len(ref_chars) - n + 1):
        key = "".join(c["ch"] for c in ref_chars[i:i + n])
        hits = index.get(key)
        if not hits or len(hits) > 12:      # 满片都有的台词不投票，纯噪声
            continue
        rt = ref_chars[i]["t"]
        for st in hits:
            votes.setdefault((round(rt, 1), round(st, 1)), 0)
            votes[(round(rt, 1), round(st, 1))] += 1
    pairs = sorted(k for k, v in votes.items() if v >= ANCHOR_MIN_VOTES)
    if not pairs:
        return []
    # 最长递增子序列（按素材时间），把多义投票压成一条剪辑链
    best, back = [], []
    for i, (_, st) in enumerate(pairs):
        pick, blen = -1, 0
        for j in range(i):
            if pairs[j][1] <= st and best[j] > blen:
                pick, blen = j, best[j]
        best.append(blen + 1)
        back.append(pick)
    tail = max(range(len(best)), key=lambda i: best[i])
    chain = []
    while tail >= 0:
        chain.append(pairs[tail])
        tail = back[tail]
    return list(reversed(chain))


def anchor_window(anchors: list, shot: tuple) -> tuple:
    """锚点表里查这一镜的素材搜索窗。查不到返回 None（全片搜）。

    窗宽随「离最近锚点多远」变宽：锚点之间的插值假的是「两边同速推进」，可爆款恰恰
    在锚点之间做了删剪，离锚点越远误差越大。固定 ±6s 时两头都不对——
    台词密的片子（r1_030 每镜都有锚点）放宽到 ±12s 就会把 12s 外那个分数更低的错解放进来；
    台词稀的片子（r1_011 全片只 7 个锚点，最近锚点隔 37s）±6s 又把真解关在窗外。
    """
    if not anchors:
        return None
    before = [a for a in anchors if a[0] <= shot[1]]
    after = [a for a in anchors if a[0] >= shot[0]]
    if not before and not after:
        return None
    gap_lo = abs(shot[0] - before[-1][0]) if before else abs(after[0][0] - shot[1])
    gap_hi = abs(after[0][0] - shot[1]) if after else abs(shot[0] - before[-1][0])
    lo = (before[-1][1] + (shot[0] - before[-1][0])) if before else after[0][1]
    hi = (after[0][1] + (shot[1] - after[0][0])) if after else before[-1][1]
    if lo > hi:
        lo, hi = hi, lo
        gap_lo, gap_hi = gap_hi, gap_lo
    return (max(0.0, lo - ANCHOR_PAD - ANCHOR_DRIFT * gap_lo),
            hi + ANCHOR_PAD + ANCHOR_DRIFT * gap_hi)


# ---------------- 步骤 3c：ASR 也对不上才上 VLM ----------------
def _grab_frame(video: str, sec: float, dst: str) -> str:
    ret = pv._run([pv._ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", "%.3f" % max(0.0, sec), "-i", video,
                   "-frames:v", "1", "-q:v", "3", "-vf", "scale=360:-2", dst])
    if ret.returncode != 0 or not os.path.isfile(dst):
        return ""
    return dst


def locate_by_vlm(shot: tuple, ref: str, mat: dict, win: tuple, work: str) -> dict:
    """抽爆款 3 帧 + 素材窗口 3 帧，问 VLM 这段是不是同一镜、起点在哪。

    窗口必须先被邻镜或 ASR 缩到几十秒：把 59 分钟原片塞进 VLM 实测 4.6 分钟未返回。
    只让它做「是/否 + 起点秒数」的校验，不让它满片检索。
    """
    start, end = shot
    dur = max(0.4, end - start)
    lo, hi = max(0.0, win[0]), max(win[0] + 0.4, win[1])
    if hi - lo > ASR_MAX_WIN:
        return {}
    try:
        import aigc
    except Exception:
        return {}
    os.makedirs(work, exist_ok=True)
    media, labels = [], []
    for i, t in enumerate((start, (start + end) / 2.0, max(start, end - 0.05))):
        dst = os.path.join(work, "r%d.jpg" % i)
        if _grab_frame(ref, t, dst):
            media.append(dst)
            labels.append("爆款第%d帧 t=%.2fs" % (i + 1, t))
    guess = (lo + hi) / 2.0
    src_ts = (max(lo, guess - dur / 2.0), guess, min(hi, guess + dur / 2.0))
    for i, t in enumerate(src_ts):
        dst = os.path.join(work, "s%d.jpg" % i)
        if _grab_frame(mat["file"], t, dst):
            media.append(dst)
            labels.append("素材第%d帧 t=%.2fs" % (i + 1, t))
    if len(media) < 4:
        return {}
    prompt = (
        "判断爆款这三帧是不是从素材这三帧所在片段剪出来的。"
        "只回一行 JSON：{\"same\":true/false,\"src_start\":秒}。"
        "src_start 是素材里对应爆款镜头起点的绝对秒数，范围 [%.2f, %.2f]。"
        "拿不准就 same=false。帧标签：%s"
        % (lo, hi, "；".join(labels)))
    try:
        raw = aigc.understand(prompt, media=media, json_mode=True)
    except Exception as exc:
        print("  VLM 失败：%s" % str(exc)[:160], flush=True)
        return {}
    flat = " ".join((raw or "").split())
    print("  VLM 原文：%s" % flat[:240], flush=True)
    m = re.search(r"\{[^{}]*\}", flat)
    if not m:
        return {}
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return {}
    print("  VLM 解析：%s" % got, flush=True)
    if not got.get("same"):
        return {}
    try:
        origin = float(got.get("src_start"))
    except (TypeError, ValueError):
        return {}
    if origin < lo - 1.0 or origin > hi + 1.0:
        return {}
    return {"命中": True, "源起点": round(max(lo, origin), 2), "平均距离": 0.0,
            "取样帧": 0, "变速": 1.0, "定位": "vlm",
            "源文件": os.path.basename(mat["file"]),
            "_src": mat["file"], "_silent": mat["silent"]}


# ---------------- 步骤 3d：三层都对不上才 AIGC 生成 ----------------
def gen_piece(shot: tuple, ref: str, dst: str, work: str) -> dict:
    """三层定位全失败的镜头，用 AIGC 补一段。返回 {} 表示补不出来。

    这是最后一手，代价最高也最不可控（要么风控拒、要么演出来的和原片不是一回事），
    所以必须记在 EDL 里、和裁出来的镜头区分开——「用户素材为主、AI 补片为辅」。
    先让 VLM 看爆款这一镜写提示词，再交 seedance 生成，最后裁回原时长。
    """
    start, end = shot
    dur = max(0.4, end - start)
    try:
        import aigc
        import storage
    except Exception:
        return {}
    os.makedirs(work, exist_ok=True)
    frames = []
    for i, t in enumerate((start, (start + end) / 2.0, max(start, end - 0.05))):
        got = _grab_frame(ref, t, os.path.join(work, "g%d.jpg" % i))
        if got:
            frames.append(got)
    if not frames:
        return {}
    try:
        prompt = aigc.understand(
            "用一句中文描述这几帧构成的镜头，写成视频生成提示词："
            "主体、动作、景别、机位、光线、场景。不要写文字/字幕/水印，不要解释。",
            media=frames)
        prompt = prompt.strip()[:300]
        # 参考图必须是公网 URL（gen_video 不吃本地路径），短剧帧又全是真人脸、
        # 容易被风控按 40000002 拒——所以参考图生成失败就退回纯文生，宁可像得少一点也要出片。
        url = ""
        try:
            ref_url = storage.upload(frames[0])
            url = aigc.gen_video(prompt, ref_images=[ref_url],
                                 duration_sec=max(4, int(dur + 0.99)))
        except Exception as exc:
            print("  AIGC 参考图生成被拒，退纯文生：%s" % str(exc)[:120], flush=True)
        if not url:
            url = aigc.gen_video(prompt, duration_sec=max(4, int(dur + 0.99)))
    except Exception as exc:
        print("  AIGC 失败：%s" % str(exc)[:160], flush=True)
        return {}
    raw = os.path.join(work, os.path.basename(dst).replace(".mp4", "_raw.mp4"))
    try:
        storage.download(url, raw)          # 走 requests：ffmpeg 直接吃这个签名 URL 拉不下来
    except Exception as exc:
        print("  AIGC 产物下载失败：%s" % str(exc)[:120], flush=True)
        return {}
    if not os.path.isfile(raw) or os.path.getsize(raw) < 1024:
        print("  AIGC 产物为空", flush=True)
        return {}
    out = cut_piece(raw, 0.0, dur, dst, True)      # 生成片一律静音，音轨由外部统一铺
    out["提示词"] = prompt
    return out


# ---------------- 步骤 4：裁剪 ----------------
def _has_audio(path: str) -> bool:
    return "Audio:" in pv._run([pv._ffmpeg(), "-hide_banner", "-i", path]).stderr


def cut_piece(src: str, start: float, dur: float, dst: str, silent: bool,
              speed: float = 1.0) -> dict:
    """从素材裁一段并统一到成片画布。画布参数与主链路共用 produce_video 的常量，
    这样重剪产物能直接和主链路的段落一起 concat。

    speed = 源时长/爆款时长：爆款把这段放快了就 >1。此时必须从素材多取 dur*speed 秒再
    压回 dur 秒，否则这一镜会比爆款长，后面所有镜头的节奏跟着错位。
    """
    take = max(0.04, dur * speed)
    vf = []
    if abs(speed - 1.0) > 0.01:
        vf.append("setpts=PTS/%.6f" % speed)
    vf.append("scale=%d:%d:force_original_aspect_ratio=decrease,"
              "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=%d"
              % (pv.TARGET_W, pv.TARGET_H, pv.TARGET_W, pv.TARGET_H, pv.TARGET_FPS))
    cmd = [pv._ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "%.3f" % max(0.0, start), "-t", "%.3f" % take, "-i", src]
    if silent:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=%d" % pv.TARGET_AR, "-shortest"]
    cmd += ["-map", "0:v:0", "-map", "1:a:0" if silent else "0:a:0", "-vf", ",".join(vf)]
    if not silent and abs(speed - 1.0) > 0.01:
        cmd += ["-af", "atempo=%.6f" % min(2.0, max(0.5, speed))]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p"]
    cmd += pv.aac_args() + [dst]
    ret = pv._run(cmd)
    if ret.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError("裁剪失败：%s" % ret.stderr[-300:])
    return {"文件": dst, "时长秒": round(pv._duration(dst), 2)}


# ---------------- 步骤 5：拼接 ----------------
def concat(files: list, outdir: str) -> str:
    """流复制拼接。各片都是 cut_piece 出的同参数片，不需要再归一化一遍。
    list.txt 里写绝对路径：concat demuxer 把相对路径按 list 文件所在目录再解一次，
    写相对路径会拼成 pieces/pieces/xxx.mp4。"""
    listfile = os.path.join(outdir, "pieces", "list.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for path in files:
            fh.write("file '%s'\n" % os.path.abspath(path).replace("'", "'\\''"))
    final = os.path.join(outdir, "final.mp4")
    ret = pv._run([pv._ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                   "-f", "concat", "-safe", "0", "-i", listfile,
                   "-c", "copy", "-movflags", "+faststart", final])
    if ret.returncode != 0 or not os.path.isfile(final):
        raise RuntimeError("拼接失败：%s" % ret.stderr[-300:])
    return final


# ---------------- 编排 ----------------
def recut(ref: str, materials: list, outdir: str, mute: bool = False,
          cache_dir: str = "", use_asr: bool = True, use_vlm: bool = True,
          use_aigc: bool = False) -> dict:
    """纯重剪主流程：拆爆款镜头 → 素材指纹定位 → 裁剪拼接成片。

    ref 是爆款参考片，materials 是用户素材（可多条）；定位三层兜底：
    hash 窗口对齐 → ASR 锚点 → VLM 校验，都失败且开 use_aigc 才生成补片。
    返回统计与逐镜明细的 dict，供 to_markdown 渲染。
    """
    os.makedirs(os.path.join(outdir, "pieces"), exist_ok=True)
    clock = time.time()

    # 共同长宽比 = 这一组内容区里最窄的：信箱对信箱不裁，全幅对中心裁只裁全幅两侧。
    ars = []
    for path in [ref] + list(materials):
        box = content_box(path)
        crop = _margin_crop(box)
        vw, vh = _video_wh(path)
        ars.append((crop[2] * vw) / max(1e-6, crop[3] * vh))
    target_ar = min(ars) if ars else 0.0

    ref_times, ref_hashes, ref_box = decode_hashes(ref, REF_FPS, cache_dir, target_ar)
    shots = split_shots(ref_times, ref_hashes, REF_FPS)
    ref_dur = float(ref_times[-1]) + 1.0 / REF_FPS
    print("爆款 %s：%.1fs，内容区 %s，拆出 %d 个镜头（%.0fs）"
          % (os.path.basename(ref), ref_dur, _box_text(ref_box), len(shots),
             time.time() - clock), flush=True)

    index = []
    for path in materials:
        mark = time.time()
        times, hashes, box = decode_hashes(path, INDEX_FPS, cache_dir, target_ar)
        # 有没有音轨探一次就够，不要每裁一镜探一遍（一条素材要裁几十镜）
        index.append({"file": path, "times": times, "hashes": hashes,
                      "silent": mute or not _has_audio(path),
                      "asr_cache": cache_dir})
        print("素材 %s：%.1fs，内容区 %s，指纹 %d 帧（%.0fs）"
              % (os.path.basename(path), float(times[-1]), _box_text(box), len(hashes),
                 time.time() - mark), flush=True)

    def place(shot):
        """在所有素材上定位一镜，返回最好的那条。有台词锚点就只搜锚点附近。"""
        best = None
        for mat in index:
            win = anchor_window(mat.get("anchors") or [], shot)
            got = locate(shot, ref_times, ref_hashes, mat["times"], mat["hashes"], window=win)
            if win is not None and got["平均距离"] > ANCHOR_TRUST:
                # 锚点窗里只勉强对上（或没对上），可能是锚点本身偏了：放开成全片再试一次。
                # 但全片解必须明显更好才采纳——差不多的分数下宁可信锚点：
                # 锚点带着时序约束，全片搜在同场景反复出现的短剧里很容易捡到分数略低的错帧
                # （r1_030 镜35 全片 4.8 分那一解在时间线上是往回跳的，锚点 6.1 分才接得上前后镜）。
                wide = locate(shot, ref_times, ref_hashes, mat["times"], mat["hashes"])
                if wide["平均距离"] + ANCHOR_MARGIN < got["平均距离"]:
                    got = wide
            got["源文件"] = os.path.basename(mat["file"])
            got["_src"] = mat["file"]
            got["_silent"] = mat["silent"]
            if best is None or (got["命中"], -got["平均距离"]) > (best["命中"], -best["平均距离"]):
                best = got
        return best

    # ---- 步骤 3a：先跑 ASR，用台词把每个镜头的搜索范围锚到素材的某一段 ----
    # 顺序是「ASR 先、哈希后」：台词是全片唯一的、抗一切画面改动（改分辨率/加字幕/贴纸/
    # 变速都不动台词）的信号，先用它把搜索范围从整条原片缩到锚点附近几十秒，哈希再在窄窗里定精确帧。
    # 反过来（哈希先）等于让哈希在 8000 帧里赌运气：r1_030 镜35 全片搜到的是一个分数更低
    # （4.8 vs 6.1）但在时间线上往回跳的错帧，加了锚点才接得上前后镜。
    # 代价：素材整片转写在无 GPU 机器上约 0.7x 实时（34 分钟原片实测 24 分钟）。但它按 30s
    # 分块落盘，一条原片剪出的所有爆款共用同一份，所以是一次性投入；缓存命中后每条爆款只多几十秒。
    asr_cache = os.path.join(cache_dir or os.path.join(outdir, ".cache"), "asr")
    for mat in index:
        mat["asr_cache"] = asr_cache
        mat["anchors"] = []
    if use_asr and _asr_ready():
        mark = time.time()
        ref_chars = asr_stream(ref, ref_dur, asr_cache, "爆款")
        print("爆款台词 %d 字（%.0fs）" % (len(ref_chars), time.time() - mark), flush=True)
        for mat in index:
            mark = time.time()
            src_chars = asr_stream(mat["file"], float(mat["times"][-1]), asr_cache,
                                   os.path.basename(mat["file"])[:12])
            mat["anchors"] = anchor_map(ref_chars, src_chars)
            print("素材 %s 台词 %d 字，台词锚点 %d 个（%.0fs）"
                  % (os.path.basename(mat["file"]), len(src_chars), len(mat["anchors"]),
                     time.time() - mark), flush=True)
    elif use_asr:
        print("本机没有可用的 Qwen3-ASR 环境，跳过台词锚点，直接哈希全片搜", flush=True)

    def neighbor_win(shot, mat):
        lo = hi = None
        name = os.path.basename(mat["file"])
        for row in rows:
            if not row.get("产物"):
                continue
            if row.get("源文件") is not name and not (row.get("源文件") == name):
                continue
            src = row["源起点"]
            if row["爆款止"] <= shot[0] + 0.05:
                lo = src + row["时长秒"] * row.get("变速", 1.0)
            elif row["爆款起"] >= shot[1] - 0.05 and hi is None:
                hi = src
        if lo is None and hi is None:
            return None
        if lo is None:
            lo = max(0.0, hi - shot[1] + shot[0] - ASR_PAD)
        if hi is None:
            hi = lo + (shot[1] - shot[0]) * 2.0 + ASR_PAD
        return (max(0.0, lo - ASR_PAD), hi + ASR_PAD)

    def rescue(shot, hashed):
        """哈希对不上：先按台词直接定位，再让 VLM 校验候选窗。"""
        query = ""
        if use_asr and _asr_ready():
            q_items = _asr_transcribe(ref, shot[0], max(0.4, shot[1] - shot[0]), asr_cache)
            query = _asr_clean("".join(str(it.get("text") or "") for it in q_items))
            if query:
                print("  ASR 查询「%s」" % query[:24], flush=True)
        for mat in index:
            # 窗口来源两条：台词锚点（全局，最可靠）和已命中的邻镜（局部）
            win = anchor_window(mat.get("anchors") or [], shot) or neighbor_win(shot, mat)
            if win is None:
                continue
            if use_asr and len(query) >= ASR_MIN_CHARS:
                got = locate_by_asr(shot, query, mat, win)
                if got:
                    return got
            if use_vlm and rescue.vlm_left > 0:
                rescue.vlm_left -= 1
                print("  VLM 校验 素材窗 %.1f–%.1fs（剩余 %d 次）" % (win[0], win[1], rescue.vlm_left), flush=True)
                got = locate_by_vlm(shot, ref, mat, win, os.path.join(outdir, "pieces", "_vlm"))
                if got:
                    return got
        return hashed
    rescue.vlm_left = VLM_MAX_SHOTS if use_vlm else 0
    rescue.aigc_left = AIGC_MAX_SHOTS if use_aigc else 0

    work, rows, pieces, no = list(shots), [], [], 0
    splits_left = len(shots)                 # 每条原硬切最多再补一刀，挡住叠化误切连环拆
    while work:
        shot = work.pop(0)
        dur = shot[1] - shot[0]
        best = place(shot)
        if not best["命中"] and splits_left > 0:
            cut = soft_cut_at(shot, ref_times, ref_hashes, REF_FPS)
            if cut is not None:
                # 这一镜其实是两段源画面被叠化并在一起。切开后各自再定位。
                splits_left -= 1
                work.insert(0, (cut, shot[1]))
                work.insert(0, (shot[0], cut))
                print("  软切 %.2f–%.2f @%.2fs（平均距离 %.1f，再拆）"
                      % (shot[0], shot[1], cut, best["平均距离"]), flush=True)
                continue
        if not best["命中"]:
            rescued = rescue(shot, best)
            if rescued.get("命中"):
                best = rescued
        no += 1
        made = None
        if not best["命中"] and use_aigc and rescue.aigc_left > 0:
            rescue.aigc_left -= 1
            print("  AIGC 生成（三层都对不上，剩余 %d 次）" % rescue.aigc_left, flush=True)
            made = gen_piece(shot, ref, os.path.join(outdir, "pieces", "s%03d.mp4" % no),
                             os.path.join(outdir, "pieces", "_gen")) or None
        row = {"镜号": no, "爆款起": round(shot[0], 2), "爆款止": round(shot[1], 2),
               "时长秒": round(dur, 2),
               **{k: v for k, v in best.items() if not k.startswith("_")}}
        if best["命中"]:
            out = cut_piece(best["_src"], best["源起点"], dur,
                            os.path.join(outdir, "pieces", "s%03d.mp4" % no), best["_silent"],
                            speed=best["变速"])
            row.update({"产物": os.path.basename(out["文件"]), "产物时长秒": out["时长秒"]})
            pieces.append(out["文件"])
        elif made:
            row.update({"产物": os.path.basename(made["文件"]), "产物时长秒": made["时长秒"],
                        "定位": "aigc", "源文件": "AI生成", "提示词": made.get("提示词", "")})
            pieces.append(made["文件"])
        else:
            row["产物"] = ""
        rows.append(row)
        how = row.get("定位") or "hash"
        if row["产物"] and how == "aigc":
            print("  镜%02d %5.2fs → AI 生成 [aigc]「%s」"
                  % (no, dur, row.get("提示词", "")[:40]), flush=True)
        else:
            print("  镜%02d %5.2fs → %s"
                  % (no, dur, "%s@%.2fs 距离%.1f%s%s"
                     % (row["源文件"], row["源起点"], row["平均距离"],
                        "" if row["变速"] == 1.0 else " 变速%.2fx" % row["变速"],
                        "" if how == "hash" else " [%s]" % how)
                     if row["产物"] else "未命中（平均距离 %.1f）" % row["平均距离"]), flush=True)

    hits = [r for r in rows if r["产物"]]
    matched = [r for r in hits if r.get("定位") != "aigc"]   # AI 补的不算「找到素材」
    result = {"爆款": os.path.basename(ref), "爆款时长秒": round(ref_dur, 2),
              "素材": [os.path.basename(m) for m in materials],
              "镜头数": len(rows), "命中数": len(matched),
              "命中率": round(len(matched) / max(1, len(rows)), 3),
              "覆盖时长比": round(sum(r["时长秒"] for r in hits) / max(0.1, ref_dur), 3),
              "变速镜数": sum(1 for r in hits if r["变速"] != 1.0),
              "锚点镜数": sum(1 for r in hits if r.get("定位") == "hash+asr锚"),
              "ASR镜数": sum(1 for r in hits if r.get("定位") == "asr"),
              "VLM镜数": sum(1 for r in hits if r.get("定位") == "vlm"),
              "AIGC镜数": sum(1 for r in hits if r.get("定位") == "aigc"),
              "耗时秒": round(time.time() - clock, 1), "分镜": rows}
    # 一个镜头都没命中也要先把 EDL 和报告落盘再报错：这份逐镜距离表就是定位失败原因的唯一线索
    # （比如几何没对齐时距离会整体停在 20 上下，和真的换了素材完全是两种分布）。
    if pieces:
        result["成片"] = concat(pieces, outdir)
        result["成片时长秒"] = round(pv._duration(result["成片"]), 2)
    with open(os.path.join(outdir, "edl.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(to_markdown(result))
    if not pieces:
        raise RuntimeError("没有一个镜头定位成功，不出片（逐镜距离见 %s）"
                           % os.path.join(outdir, "report.md"))
    print("素材命中 %d/%d 镜%s，覆盖爆款时长 %.0f%%，成片 %.1fs → %s"
          % (len(matched), len(rows),
             ("，AI 补 %d 镜" % result["AIGC镜数"]) if result["AIGC镜数"] else "",
             result["覆盖时长比"] * 100,
             result["成片时长秒"], result["成片"]), flush=True)
    return result


def to_markdown(result: dict) -> str:
    """把 recut() 返回的结果渲染成 Markdown 报告（概览 + 逐镜明细表）。"""
    lines = ["# 纯重剪：%s" % result["爆款"], "",
             "- 素材：%s" % "、".join(result["素材"]),
             "- 爆款 %.1fs / %d 镜，命中 %d 镜（%.0f%%），覆盖时长 %.0f%%"
             % (result["爆款时长秒"], result["镜头数"], result["命中数"],
                result["命中率"] * 100, result["覆盖时长比"] * 100),
             "- 成片 %.1fs，耗时 %.0fs；变速 %d 镜"
             % (result.get("成片时长秒", 0), result["耗时秒"], result.get("变速镜数", 0)),
             "- 定位来源：台词锚点内命中 %d 镜，ASR 直接定位 %d 镜，VLM 校验 %d 镜，AIGC 生成 %d 镜"
             % (result.get("锚点镜数", 0), result.get("ASR镜数", 0),
                result.get("VLM镜数", 0), result.get("AIGC镜数", 0)), "",
             "| 镜 | 爆款区间 | 时长 | 素材位置 | 变速 | 定位 | 平均汉明距离 | 产物 |",
             "|---|---|---|---|---|---|---|---|"]
    for row in result["分镜"]:
        if not row["产物"]:
            where = "未命中"
        elif row.get("定位") == "aigc":
            where = "AI生成"
        else:
            where = "%s@%.2fs" % (row["源文件"], row["源起点"])
        lines.append("| %d | %.2f–%.2f | %.2fs | %s | %.2fx | %s | %.2f | %s |"
                     % (row["镜号"], row["爆款起"], row["爆款止"], row["时长秒"], where,
                        row["变速"], row.get("定位") or "hash",
                        row["平均距离"], row["产物"] or "—"))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """命令行入口：解析参数并调用 recut()，返回进程退出码。"""
    ap = argparse.ArgumentParser(description="纯重剪：用用户素材还原爆款的剪辑，零模型调用")
    ap.add_argument("reference", help="爆款参考片")
    ap.add_argument("materials", nargs="+", help="用户素材（可多条，会一起建指纹索引）")
    ap.add_argument("--outdir", default="", help="产物目录，默认 output/recut/<爆款名>")
    ap.add_argument("--mute", action="store_true", help="成片静音（默认带素材原声）")
    ap.add_argument("--cache", default=".recut_cache",
                    help="指纹缓存目录，空字符串关闭（同一条素材复用时省掉重复解码）")
    ap.add_argument("--no-asr", action="store_true", help="关掉哈希未命中后的 ASR 补定位")
    ap.add_argument("--no-vlm", action="store_true", help="关掉 ASR 也对不上后的 VLM 校验")
    ap.add_argument("--aigc", action="store_true",
                    help="三层都对不上的镜头用 AIGC 生成补上（默认关：这条链路的定位是纯重剪）")
    args = ap.parse_args(argv)
    for path in [args.reference] + args.materials:
        if not os.path.isfile(path):
            raise SystemExit("找不到文件：%s" % path)
    outdir = args.outdir or os.path.join("output", "recut",
                                         os.path.splitext(os.path.basename(args.reference))[0])
    recut(args.reference, args.materials, outdir, mute=args.mute, cache_dir=args.cache,
          use_asr=not args.no_asr, use_vlm=not args.no_vlm, use_aigc=args.aigc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
