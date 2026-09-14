"""任务目录与素材登记：一个任务 = {任务根目录}/{task_id}/。

参数、中间产物、状态、日志全部落在任务目录里，所有步骤都通过这里读写任务记录。
任务根目录默认 output/tasks（命令行），Web 服务用 VF_TASKS_DIR 指到 output/web/tasks。
"""
import glob
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid

import config  # pyright: ignore[reportImplicitRelativeImport]

# 任务根目录。默认 output/tasks（命令行与实验脚本用）；Web 服务在 import 前把
# VF_TASKS_DIR 指到 output/web/tasks，让前端的输入、中间结果、成片单独成一片，
# 不跟命令行跑的实验任务混在一起。
CLI_TASKS_DIR = os.path.abspath(os.path.join(config.OUTPUT_DIR, "tasks"))
TASKS_DIR = os.path.abspath(os.getenv("VF_TASKS_DIR", "").strip() or CLI_TASKS_DIR)
# 素材复用要覆盖两个根：前端应该能直接用命令行任务传过的素材，反之亦然。
TASK_ROOTS = tuple(dict.fromkeys((TASKS_DIR, CLI_TASKS_DIR)))

# 可复用素材库：manifest.json 负责中文展示名和素材组，媒体文件可按自己的归档方式存放。
# 历史任务上传过的文件也能被前端直接勾选，不必重复上传大视频。两边共用一个库。
LIBRARY_DIR = os.path.join(config.OUTPUT_DIR, "library")
LIBRARY_MANIFEST = os.path.join(LIBRARY_DIR, "manifest.json")

# 并行阶段（pipeline.STAGES）多线程共享同一个任务 rec：写状态、追日志、落盘 task.json
# 都必须串行，否则 json.dump 迭代到一半 dict 变了会直接炸
_LOCK = threading.RLock()
_MD5_CACHE = {}
# 被要求终止的任务。Python 线程杀不掉，只能协作式退出：标记进来之后，任何一次落盘都
# 直接抛 TaskCancelled，跑着的线程就在下一个写盘点退出。
# 这一步是删除运行中任务的前提——_p() 会 makedirs，如果线程还在写盘，目录删掉又会被
# 它重建出来，任务就删不掉。
_CANCELLED = set()


class TaskCancelled(RuntimeError):
    """任务被手动终止，用来打断流水线里任意位置的后续写盘。"""


def cancel_task(task_id: str) -> None:
    """标记任务终止；运行线程会在下一次 save 时抛 TaskCancelled 退出。"""
    _CANCELLED.add(task_id)


def uncancel_task(task_id: str) -> None:
    """清掉终止标记（任务已删完，或用户改主意要继续用这个 id）。"""
    _CANCELLED.discard(task_id)


def is_cancelled(task_id: str) -> bool:
    return task_id in _CANCELLED


DEFAULT_OPTIONS = {
    "copy_bgm": True,
    # 字幕默认跟随参考片：参考片画面没字的，复刻出来也不该凭空多一层字。
    # None=跟随参考片 / False=完全不烧 / "force"=命令行强制烧
    "copy_subtitles": None,
    "rewrite_mode": "creative",   # creative（默认）=允许大模型改写剧本 | strict=只替换人物/商品/场景
    # 整片叙事审查（含自动修补与复审）不通过就停在 script 步，不往下出设定图、不分段、不出片。
    # 这道闸刻意放在任何生成动作之前：等成片出来再发现整片看不懂，钱和时间都已经花掉了。
    # 关掉它就退回「只告警、照旧往下跑」。
    "narrative_gate": True,
    # 用户素材一律静音后用克隆音色重配：默认 True（整片统一一把嗓子，音色最整齐，
    # 也避开素材原声忽大忽小、混入环境音的问题）。关成 False 退回按「用户切片处理」表
    # 逐条判，口播内容对得上又没 BGM 的切片保留原声。开着的代价有两条：丢掉素材原声的
    # 现场感；素材里提不出音色基准（或 TTS 后端没就绪）时，这些本来能用原声的切片会变哑片。
    "dub_all_cuts": True,
    "seg_workers": 5,
}


# ---------------- 任务存取 ----------------
_ID_RE = re.compile(r"^[\w-]+$")


def task_dir(task_id: str) -> str:
    """任务目录。这里是 task_id 的唯一校验点，_p/_d/load/save/delete_task 全走它。

    合法 task_id 只有两种来源（见 create_task）：_safe_slug 出来的「中英文数字 _」，
    或者「时间戳-哈希」。`\\w` 与 `-` 覆盖这两种，同时挡住 `.` 与 `/`：
    task_id 传 ".." 时 os.path.join(TASKS_DIR, "..") 经 realpath 会跳到任务根目录的父目录，
    /files 就能列出跨全部任务的产物、/file 能读出来（实测 1169 个文件），
    上传接口的 _p() 还会 makedirs 到任务根之外去。
    """
    if not task_id or not _ID_RE.match(str(task_id)):
        raise ValueError("非法任务 ID：%r" % (task_id,))
    return os.path.join(TASKS_DIR, task_id)


def _p(task_id: str, *parts: str) -> str:
    """任务目录下的绝对路径，父目录自动创建。"""
    path = os.path.join(task_dir(task_id), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _rel(task_id: str, path: str) -> str:
    return os.path.relpath(path, task_dir(task_id))


def _d(task_id: str, *parts: str) -> str:
    """任务目录下的子目录，自动创建。"""
    path = os.path.join(task_dir(task_id), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def load(task_id: str) -> dict:
    with open(os.path.join(task_dir(task_id), "task.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    rec["title_auto"] = _is_auto_title(rec)
    return rec


def save(rec: dict) -> dict:
    with _LOCK:
        # 已被终止的任务不再落盘：_p() 会 makedirs，删掉的目录会被重新建出来
        if rec.get("task_id") in _CANCELLED:
            raise TaskCancelled("任务已终止：%s" % rec.get("task_id"))
        rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = _p(rec["task_id"], "task.json")
        tmp = path + ".tmp"                  # 前端在轮询，写一半的 json 不能被读到
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return rec


def log(rec: dict, msg: str) -> None:
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    with _LOCK:
        rec.setdefault("logs", []).append(line)
        rec["logs"] = rec["logs"][-400:]
    print("[%s] %s" % (rec["task_id"], msg), flush=True)


def _safe_slug(text: str) -> str:
    """目录名安全化：保留中英文与数字，其余字符（路径分隔符、空格、标点）替换成下划线。"""
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:80]


def _default_title(product: dict = None, created_at: str = "") -> str:
    """默认任务名：「MM-DD HH:MM · 商品名」。

    带时间戳是为了在任务列表里一眼认出是哪次跑的；用 created_at 而不是当前时间，
    这样事后清空名字重新生成，时间戳仍指向建任务的那一刻。
    """
    name = ((product or {}).get("name") or "").strip()
    stamp = (created_at or time.strftime("%Y-%m-%d %H:%M:%S"))[5:16]
    return "%s · %s" % (stamp, name) if name else "%s 复刻任务" % stamp


def _is_auto_title(rec: dict) -> bool:
    """当前任务名是不是自动生成的（用户没手动改过）。

    分叉出新任务时要靠这个判断名字能不能照搬：自动名里嵌着建任务的时刻，照搬过去新任务
    显示的还是老时间戳；用户手起的名字才该跟着走。老 task.json 没有这个字段，用默认名
    反推一次。
    """
    if "title_auto" in rec:
        return bool(rec["title_auto"])
    return (rec.get("title") or "") == _default_title(rec.get("product"), rec.get("created_at"))


def create_task(product: dict = None, options: dict = None, title: str = "",
                slug: str = "") -> dict:
    """建任务并落盘。product 可以先留空，上传素材后再补。

    slug 给定时用它当 task_id（目录名可读，如 20260818_17_参考苹果笔记本生成小米手机），
    重名自动追加 _2/_3；不给则退回「时间戳-哈希」老格式（Web 端建任务时还没有用例信息）。"""
    if slug and _safe_slug(slug):
        base = _safe_slug(slug)
        task_id, n = base, 2
        while os.path.isdir(task_dir(task_id)):
            task_id = "%s_%d" % (base, n)
            n += 1
    else:
        task_id = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:4])
    opts = dict(DEFAULT_OPTIONS)
    opts.update({k: v for k, v in (options or {}).items() if k in DEFAULT_OPTIONS})
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    rec = {
        "task_id": task_id,
        "title": title.strip() if (title or "").strip() else _default_title(product, created_at),
        # 名字是自动生成还是用户手起的：分叉新任务时决定要不要把名字带过去
        "title_auto": not (title or "").strip(),
        "created_at": created_at,
        "status": "draft",
        "step": "",
        "options": opts,
        "product": {"name": "", "selling_points": [], "category": "", "description": ""},
        "inputs": {"reference_video": "", "product_images": [], "user_videos": [],
                   "person_images": [], "bgm": ""},
        "steps": {},
        "logs": [],
    }
    rec["product"].update({k: v for k, v in (product or {}).items() if k in rec["product"]})
    os.makedirs(_p(task_id, "assets", "uploads"), exist_ok=True)
    log(rec, "任务已创建")
    return save(rec)


def update_task(task_id: str, product: dict = None, options: dict = None,
                title: str = None) -> dict:
    """改商品信息 / 复刻参数 / 任务名。

    title 传 None 表示不动；传空串表示「恢复默认名」，按 created_at + 当前商品名重算——
    所以前端把输入框清空就能回到自动命名，而不是留下一个空标题。
    """
    with _LOCK:                              # load→改→save 必须整段串行，见 register_input
        rec = load(task_id)
        was_auto, old_title = _is_auto_title(rec), rec.get("title") or ""
        if product:
            rec["product"].update({k: v for k, v in product.items() if k in rec["product"]})
        if options:
            rec["options"].update({k: v for k, v in options.items() if k in DEFAULT_OPTIONS})
        if title is not None:
            text = title.strip()
            rec["title"] = text or _default_title(rec["product"], rec.get("created_at"))
            # 前端存商品信息时会把输入框里的名字原样回传，自动名回传回来仍然算自动名：
            # 否则它一被当成用户手起的名字，下次分叉就把带老时间戳的名字带到新任务上。
            rec["title_auto"] = not text or (was_auto and text == old_title)
        return save(rec)


def list_tasks(limit: int = 50) -> list:
    """任务摘要列表，最近的在前。"""
    out = []
    for path in sorted(glob.glob(os.path.join(TASKS_DIR, "*", "task.json")), reverse=True)[:limit]:
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({k: rec.get(k) for k in ("task_id", "title", "status", "step", "created_at")})
    return out


# ---------------- 素材登记 ----------------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
KIND_EXTS = {"reference_video": VIDEO_EXTS, "user_videos": VIDEO_EXTS,
             "product_images": IMAGE_EXTS, "person_images": IMAGE_EXTS, "bgm": AUDIO_EXTS}
# 哪些输入位收多个文件。跟 create_task 里 inputs 的初始形状一一对应：
# 单值位登记成 list 的话，pipeline.step_ingest 的 os.path.isfile(参考片) 会直接判假。
KIND_MULTI = {"reference_video": False, "user_videos": True, "product_images": True,
              "person_images": True, "bgm": False}


def register_input(task_id: str, kind: str, path: str, action: str = "上传素材") -> dict:
    """把已经落到任务目录的上传文件登记进 inputs。kind 见 KIND_EXTS。

    action 只进服务端 stdout，不写任务日志：挑素材时经常反复换，一条条记进日志会把
    页面上的日志刷满，还看不出最后到底哪几个文件真进了这次复刻。开跑时由
    pipeline._log_inputs 一次性把最终清单写进日志。

    load→改→save 整段在 _LOCK 里（RLock，save 自己再拿一次没关系）：前端一次拖进多个
    文件是并发请求，各自读到同一份 rec、各自 append 自己那条、后写的覆盖先写的，
    结果少登记几条素材。
    """
    if kind not in KIND_EXTS:
        raise ValueError("不支持的素材类型：%s" % kind)
    if not os.path.isfile(path):
        raise ValueError("文件不存在：%s" % path)
    if not path.lower().endswith(KIND_EXTS[kind]):
        raise ValueError("%s 只接受 %s" % (kind, "/".join(KIND_EXTS[kind])))
    with _LOCK:
        rec = load(task_id)
        # 老任务缺新输入位；reference_video 与 bgm 是单值位，别建成 list
        rec["inputs"].setdefault(kind, [] if KIND_MULTI[kind] else "")
        if isinstance(rec["inputs"][kind], list):
            if path not in rec["inputs"][kind]:
                rec["inputs"][kind].append(path)
        else:
            rec["inputs"][kind] = path
        print("[%s] %s %s：%s" % (task_id, action, kind, os.path.basename(path)), flush=True)
        return save(rec)


def remove_input(task_id: str, kind: str, path: str = "") -> dict:
    """摘掉任务里的一个素材；path 传空表示清空这一类（前端素材槽的「清空」按钮）。

    只改登记，不动 assets/uploads 里的文件：同一个文件可能被别的任务按路径引用着。
    """
    with _LOCK:
        rec = load(task_id)
        cur = rec["inputs"].get(kind)
        if isinstance(cur, list):
            rec["inputs"][kind] = [] if not path else [p for p in cur if p != path]
        elif cur is not None and (not path or cur == path):
            rec["inputs"][kind] = ""
        return save(rec)


# ---------------- 素材复用 ----------------
def _under(path: str, root: str) -> bool:
    real, root = os.path.realpath(path), os.path.realpath(root)
    return real == root or real.startswith(root + os.sep)


def delete_task(task_id: str) -> str:
    """删掉整个任务目录（素材、中间结果、成片一起没）。返回被删的目录。

    只允许删任务根目录的直接子目录：task_id 里带 ../ 或绝对路径时 realpath 会跳出
    TASKS_DIR，这里直接拒掉，免得一个删除接口变成任意目录删除。
    """
    target = os.path.realpath(task_dir(task_id))
    if os.path.dirname(target) != os.path.realpath(TASKS_DIR):
        raise ValueError("非法任务 ID：%s" % task_id)
    if not os.path.isdir(target):
        raise FileNotFoundError("任务不存在：%s" % task_id)
    shutil.rmtree(target)
    print("[%s] 任务已删除：%s" % (task_id, target), flush=True)
    return target


def is_reusable(path: str) -> bool:
    """路径是否落在允许复用/预览的根目录下（素材库或任务目录）。

    所有拿前端传来的路径去读盘的地方都要过这一关，否则等于开了任意文件读取。
    """
    return bool(path) and any(_under(path, root) for root in (LIBRARY_DIR,) + TASK_ROOTS)


def _media_of(name: str) -> str:
    low = name.lower()
    if low.endswith(VIDEO_EXTS):
        return "video"
    if low.endswith(IMAGE_EXTS):
        return "image"
    if low.endswith(AUDIO_EXTS):
        return "audio"
    return ""


def file_md5(path: str) -> str:
    """分块计算文件 MD5，并按文件大小和修改时间缓存，避免重复扫描大视频。"""
    stat = os.stat(path)
    cached = _MD5_CACHE.get(path)
    stamp = (stat.st_size, stat.st_mtime_ns)
    if cached and cached[:2] == stamp:
        return cached[2]
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if len(_MD5_CACHE) >= 2000:
        _MD5_CACHE.pop(next(iter(_MD5_CACHE)))
    _MD5_CACHE[path] = (stamp[0], stamp[1], value)
    return value


def find_reusable_file(md5: str, size: int = 0, media: str = "", exclude: str = "") -> str:
    """在素材库和历史任务的上传目录里按 MD5 查找同一媒体文件，命中后可直接复用而不重传。

    任务目录只翻 assets/uploads/：那里放的是任务的**输入素材**，语义上确实可以互相复用。
    整个任务目录都翻的话会命中别人的**产物**（render/final.mp4、generated/seg_xx.mp4），
    而这些正是 reset_from 会挪进 history/ 的对象——别人重跑一次，这条 inputs 就断了。
    """
    roots = tuple(dict.fromkeys((LIBRARY_DIR,) + TASK_ROOTS))
    excluded = os.path.realpath(exclude) if exclude else ""
    for root in roots:
        if not os.path.isdir(root):
            continue
        scan = [root]
        if root in TASK_ROOTS:
            scan = [os.path.join(root, d, "assets", "uploads")
                    for d in sorted(os.listdir(root))]
        for base in scan:
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, names in os.walk(base):
                for name in names:
                    path = os.path.realpath(os.path.join(dirpath, name))
                    if (path == excluded or not _media_of(name)
                            or (media and _media_of(name) != media)):
                        continue
                    try:
                        if size and os.path.getsize(path) != size:
                            continue
                        if file_md5(path) == md5:
                            return path
                    except OSError:
                        continue
    return ""


def adopt_reusable(src: str, dst: str) -> str:
    """把复用命中的文件「接管」到 dst，返回真正要登记进 inputs 的路径。

    不能直接让 inputs 指向别的任务目录里的文件：去重命中时上传接口会删掉本任务的副本，
    于是这条任务只剩一个指向任务 A 的绝对路径；任务 A 一旦被删（或 reset_from 把产物挪走），
    本任务的 step_ingest 就报「缺少爆款参考视频」，而前端素材栏还显示着那个文件。
    改成硬链接：同一个 inode，不额外占磁盘（去重的收益还在），也不受源任务生命周期影响。
    跨文件系统或不支持硬链接时退回复制——多占一份磁盘，但至少不会凭空消失。
    """
    if os.path.realpath(src) == os.path.realpath(dst):
        return dst
    if os.path.exists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return dst


def find_reusable_video(md5: str, size: int = 0, exclude: str = "") -> str:
    """兼容原视频查询入口。"""
    return find_reusable_file(md5, size, "video", exclude)


def _scan_media(root: str, limit: int) -> list:
    """递归收集一个目录下的图/视频/音频，name 用相对路径好认出子目录。"""
    out = []
    for dirpath, _dirs, names in os.walk(root):
        for name in sorted(names):
            if len(out) >= limit:
                return out
            path = os.path.join(dirpath, name)
            if not _media_of(name) or not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            out.append({"path": path, "name": os.path.relpath(path, root),
                        "media": _media_of(name), "size_mb": round(size / (1 << 20), 2)})
    return out


def _short_video_title(task_path: str, rec: dict, video_path: str) -> str:
    """从拆解摘要或文件名生成简短中文描述，绝不把时间任务号当展示名。"""
    analysis_path = os.path.join(task_path, "reference", "analysis.json")
    try:
        with open(analysis_path, encoding="utf-8") as fh:
            summary = str((json.load(fh).get("整体") or {}).get("一句话概要") or "").strip()
    except (OSError, ValueError, AttributeError):
        summary = ""
    if summary:
        summary = re.sub(r"^(本片|该片|视频|全片)?通过", "", summary)
        brief = re.split(r"[，。；！]", summary, maxsplit=1)[0].strip()
        if brief:
            return brief[:32] + ("…" if len(brief) > 32 else "")
    stem = os.path.splitext(os.path.basename(video_path))[0]
    stem = re.sub(r"^\d+[_\-\s]*", "", stem).replace("_", " ").strip()
    if re.search(r"[\u4e00-\u9fff]", stem):
        return stem[:32]
    title = str(rec.get("title") or "").strip()
    if title and not re.match(r"^\d{2}-\d{2}\s+\d{2}:\d{2}", title):
        return title[:32]
    return "待补充中文描述的爆款视频"


def _historical_hot_videos(limit: int, known_digests: set) -> list:
    """汇总历史任务参考片，按内容去重，并生成业务可读的中文名称。

    多个任务常常复用同一条参考片，素材库里也可能已经有同一个文件，只按路径去重会
    让同一条视频在爆款库里反复出现，所以这里统一按 MD5 判重。
    """
    out = []
    for root in TASK_ROOTS:
        pattern = os.path.join(root, "*", "task.json")
        for task_json in sorted(glob.glob(pattern), reverse=True):
            try:
                with open(task_json, encoding="utf-8") as fh:
                    rec = json.load(fh)
                path = os.path.realpath((rec.get("inputs") or {}).get("reference_video") or "")
                if not path or not os.path.isfile(path):
                    continue
                digest = file_md5(path)
            except (OSError, ValueError, AttributeError):
                continue
            if digest in known_digests:
                continue
            known_digests.add(digest)
            out.append({"path": path, "name": _short_video_title(os.path.dirname(task_json), rec, path),
                        "media": "video", "kind": "reference_video", "source": "已上传",
                        "category": "", "form": "",
                        "size_mb": round(os.path.getsize(path) / (1 << 20), 2)})
            if len(out) >= limit:
                return out
    return out


def _reference_group_name(task_path: str, rec: dict) -> str:
    """为历史参考素材生成清晰名称，不使用时间任务号。"""
    product = rec.get("product") or {}
    name = str(product.get("name") or "").strip()
    if not name:
        name = str(((rec.get("steps") or {}).get("product") or {}).get("product") or "").strip()
    title = str(rec.get("title") or "").strip()
    if not name and title and not re.match(r"^\d{2}-\d{2}\s+\d{2}:\d{2}", title):
        name = title
    if not name:
        try:
            with open(os.path.join(task_path, "product", "fact_card.json"), encoding="utf-8") as fh:
                name = str(json.load(fh).get("name") or "").strip()
        except (OSError, ValueError, AttributeError):
            name = ""
    return name or "待补充商品名称的参考素材"


def _reference_material_name(group: str, path: str, kind: str, index: int) -> str:
    """优先使用原文件中文名，否则用商品组名和素材类型生成中文展示名。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^\d+[_\-\s]*", "", stem).replace("_", " ").strip()
    if re.search(r"[\u4e00-\u9fff]", stem):
        return stem[:36]
    label = "商品图" if kind == "product_images" else "商品视频"
    return "%s · %s %d" % (group, label, index)


def _historical_reference_groups(limit: int, known_digests: set) -> list:
    """按任务当前登记的输入构建参考素材组，忽略上传目录中遗留的旧文件。

    组名取自任务里填写的商品名，只是当时的填写结果，同名大小写不同的要合并成一组，
    描述里也要说明来源，避免把任务填错的名字当成权威商品名。
    """
    groups, order = {}, []
    for root in TASK_ROOTS:
        for task_json in sorted(glob.glob(os.path.join(root, "*", "task.json")), reverse=True):
            try:
                with open(task_json, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            task_path = os.path.dirname(task_json)
            group_name = _reference_group_name(task_path, rec)
            materials = []
            inputs = rec.get("inputs") or {}
            for index, path in enumerate(inputs.get("product_images") or [], 1):
                path = os.path.realpath(path)
                if not _under(path, task_path):
                    continue
                try:
                    digest = file_md5(path)
                except OSError:
                    continue
                if digest in known_digests:
                    continue
                known_digests.add(digest)
                materials.append({
                    "path": path,
                    "name": _reference_material_name(group_name, path, "product_images", index),
                    "media": "image",
                    "kind": "product_images",
                    "size_mb": round(os.path.getsize(path) / (1 << 20), 2),
                })
            for index, path in enumerate(inputs.get("user_videos") or [], 1):
                path = os.path.realpath(path)
                if not _under(path, task_path) or not os.path.isfile(path):
                    continue
                try:
                    digest = file_md5(path)
                except OSError:
                    continue
                if digest in known_digests:
                    continue
                known_digests.add(digest)
                materials.append({
                    "path": path,
                    "name": _reference_material_name(group_name, path, "user_videos", index),
                    "media": "video",
                    "kind": "user_videos",
                    "size_mb": round(os.path.getsize(path) / (1 << 20), 2),
                })
            if not materials:
                continue
            key = group_name.casefold()
            if key in groups:
                groups[key]["materials"].extend(materials)
                continue
            groups[key] = {"name": group_name,
                           "description": "历史任务素材，名称取自当时任务里填写的商品名",
                           "category": "",
                           "materials": materials}
            order.append(key)
            if len(order) >= limit:
                return [groups[k] for k in order]
    return [groups[k] for k in order]


def _library_item(raw: dict, default_kind: str = "") -> dict | None:
    """把清单项解析成可安全复用的素材记录；不存在或格式不对的文件不展示。"""
    rel = str(raw.get("file") or "").strip()
    path = os.path.realpath(os.path.join(LIBRARY_DIR, rel))
    if not rel or not _under(path, LIBRARY_DIR) or not os.path.isfile(path):
        return None
    media = _media_of(path)
    kind = str(raw.get("kind") or default_kind).strip()
    if not media or (kind and (kind not in KIND_EXTS or not path.lower().endswith(KIND_EXTS[kind]))):
        return None
    return {"path": path, "name": str(raw.get("title") or os.path.basename(path)),
            "media": media, "kind": kind, "source": "素材库",
            "category": str(raw.get("category") or ""), "form": str(raw.get("form") or ""),
            "size_mb": round(os.path.getsize(path) / (1 << 20), 2)}


def _library_manifest() -> dict:
    """读取素材库清单；清单写坏时降级为空库，不阻断历史素材的复用。"""
    try:
        with open(LIBRARY_MANIFEST, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _hidden_media(manifest: dict) -> tuple:
    """manifest 的 hidden 列出的素材：暂时不想在页面上出现，但文件留着不删。

    光把清单条目删掉挡不住：爆款库还会按 MD5 把历史任务用过的参考片补进来，目录扫描
    也照样列。所以这里连内容指纹一起算出来，清单 / 历史任务 / 目录扫描三个来源统一过滤。
    路径可写相对素材库根目录的相对路径，也可以写绝对路径。
    """
    paths, digests = set(), set()
    for raw in manifest.get("hidden") or []:
        target = str(raw or "").strip()
        if not target:
            continue
        full = os.path.realpath(target if os.path.isabs(target)
                                else os.path.join(LIBRARY_DIR, target))
        if not os.path.isfile(full):
            continue
        paths.add(full)
        try:
            digests.add(file_md5(full))
        except OSError:
            continue
    return paths, digests


def list_library(limit_per_dir: int = 300) -> dict:
    """返回业务素材库和兼容历史素材。

    manifest.json 里的 hot_videos 是扁平的爆款视频库；reference_groups 是带中文名称的
    素材包；hidden 里的素材一律不出现在页面上。未配置清单时，仍保留旧目录扫描与历史
    任务素材列表。
    """
    manifest = _library_manifest()
    hidden_paths, hidden_digests = _hidden_media(manifest)

    def keep(item: dict) -> bool:
        return os.path.realpath(item.get("path") or "") not in hidden_paths

    hot_videos = []
    # 预置隐藏指纹：下面的去重判断顺手就把隐藏素材挡在外面，历史任务补进来的也一样
    hot_digests = set(hidden_digests)
    for raw in manifest.get("hot_videos") or []:
        if isinstance(raw, dict):
            item = _library_item(raw, "reference_video")
            if not item:
                continue
            try:
                digest = file_md5(item["path"])
            except OSError:
                continue
            if digest in hot_digests:
                continue
            hot_digests.add(digest)
            hot_videos.append(item)
    hot_videos.extend(_historical_hot_videos(limit_per_dir, hot_digests))
    groups = []
    known_digests = set(hidden_digests)
    for raw_group in manifest.get("reference_groups") or []:
        if not isinstance(raw_group, dict):
            continue
        materials = []
        for raw in raw_group.get("materials") or []:
            if isinstance(raw, dict):
                item = _library_item(raw)
                if item and item["kind"]:
                    # 图片和视频都按内容去重：清单里已登记的素材不要再以历史任务的
                    # 名义重复出现，否则同一个文件会带着两个不同的商品名。
                    try:
                        digest = file_md5(item["path"])
                    except OSError:
                        continue
                    if digest in known_digests:
                        continue
                    known_digests.add(digest)
                    materials.append(item)
        if materials:
            groups.append({"name": str(raw_group.get("name") or "未命名参考素材"),
                           "description": str(raw_group.get("description") or ""),
                           "category": str(raw_group.get("category") or ""),
                           "materials": materials})
    groups.extend(_historical_reference_groups(limit_per_dir, known_digests))
    cats = []
    loose = [f for f in _scan_media(LIBRARY_DIR, limit_per_dir)
             if os.sep not in f["name"] and keep(f)]
    if loose:
        cats.append({"name": "未分类", "source": "素材库", "files": loose})
    for path in sorted(glob.glob(os.path.join(LIBRARY_DIR, "*"))):
        if not os.path.isdir(path):
            continue
        files = [f for f in _scan_media(path, limit_per_dir) if keep(f)]
        if files:
            cats.append({"name": os.path.basename(path), "source": "素材库", "files": files})
    for root in TASK_ROOTS:
        for updir in sorted(glob.glob(os.path.join(root, "*", "assets", "uploads")), reverse=True):
            files = [f for f in _scan_media(updir, limit_per_dir) if keep(f)]
            if files:
                cats.append({"name": updir.split(os.sep)[-3], "source": "历史任务",
                             "files": files})
    return {"library_dir": LIBRARY_DIR, "hot_videos": hot_videos,
            "reference_groups": groups, "categories": cats}


def pick_input(task_id: str, kind: str, path: str) -> dict:
    """复用已有素材：inputs 直接指向原文件，不复制、不重传。

    路径必须落在素材库或任务目录里，否则前端可以把任意系统文件登记进任务并顺着
    产物接口读出来。
    """
    if not is_reusable(path):
        raise ValueError("只能复用素材库或历史任务目录里的文件")
    return register_input(task_id, kind, os.path.realpath(path), action="选用素材")
