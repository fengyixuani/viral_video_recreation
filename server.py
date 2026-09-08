"""ViralForge 爆款复刻 Web 服务：Flask API + 单页前端（web/index.html）。

跑法：python3 server.py（监听端口等参数在 main() 里改，或用环境变量覆盖）
  VF_WEB_HOST / VF_WEB_PORT / VF_WEB_TOKEN

安全提示：默认没有账号体系。设置 VF_WEB_TOKEN 后所有 /api 请求都要带
Authorization: Bearer <token> 或 ?token=<token>；不设置就是任何能访问该端口的人
都能建任务、消耗模型额度并读取任务产物，只适合内网/隧道访问。

接口：
  POST /api/tasks                     建任务（product + options）
  GET  /api/tasks                     任务列表
  GET  /api/tasks/<id>                任务详情（状态/步骤/日志/结果）
  PATCH/api/tasks/<id>                改商品信息、复刻参数与任务名
  DELETE /api/tasks/<id>              删除任务（连素材、中间结果、成片一起删）
  POST /api/tasks/<id>/upload         上传素材（multipart，字段 kind + file）
  POST /api/tasks/<id>/pick           复用素材库/历史任务里已有的素材（不重传）
  GET  /api/library                   可复用素材清单（按文件夹分类，含 media 类型）
  GET  /api/library/file-by-md5       上传前查询相同图片或视频
  GET  /api/library/video-by-md5      上传前查询相同视频，命中后直接复用
  GET  /api/library/file?path=        预览素材库/历史任务里的素材文件
  POST /api/tasks/<id>/run            后台跑流水线
  POST /api/tasks/<id>/reset          从某步骤重跑
  GET  /api/tasks/<id>/overview       关键信息（拆解/事实卡/剧本/分段时间线/成片）
  GET  /api/tasks/<id>/files          任务目录下所有产物（中间结果+成片）清单
  GET  /api/tasks/<id>/file?path=     取任务目录内的产物（成片、md、json）
"""
import json
import os
import re
import threading
import time

# 前端的输入、中间结果、成片单独放 output/web/tasks，跟命令行/实验跑的 output/tasks 分开。
# 必须在 import pipeline 之前设好：task_store 在 import 时就读这个变量定下任务根目录。
os.environ.setdefault("VF_TASKS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "web", "tasks"))

from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from werkzeug.serving import WSGIRequestHandler  # noqa: E402

import pipeline  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport]
import media  # noqa: E402  # pyright: ignore[reportImplicitRelativeImport]

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
# 视频首帧图缓存：放在素材库里的隐藏目录，_scan_media 不会把它当成可选素材
POSTER_DIR = os.path.join(pipeline.LIBRARY_DIR, ".posters")
TOKEN = os.getenv("VF_WEB_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.getenv("VF_MAX_UPLOAD_MB", "512"))

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * (1 << 20)

_running: "dict[str, threading.Thread]" = {}
_RUN_LOCK = threading.Lock()   # 只护 _running 的判定+登记，见 run()


class TokenSafeHandler(WSGIRequestHandler):
    """访问日志里把 ?token= 的值抹掉。

    图片和视频的 src 带不了 Authorization 头，只能把口令放在 URL 上，于是每条媒体请求
    都会把口令明文写进日志。日志被别人看到就等于口令泄露，所以落盘前先脱敏。
    """

    def log_request(self, code="-", size="-"):
        """覆写请求日志：只隐藏口令，其余保持 werkzeug 原样。"""
        safe = re.sub(r"([?&]token=)[^&\s]+", r"\1***", self.requestline)
        self.log("info", '"%s" %s %s', safe, code, size)


def _authorized() -> bool:
    """校验请求是否带有效访问口令（未设置 TOKEN 时恒放行）。"""
    if not TOKEN:
        return True
    header = request.headers.get("Authorization", "")
    supplied = header[7:].strip() if header.startswith("Bearer ") else request.args.get("token", "")
    return supplied == TOKEN


@app.before_request
def _guard():
    """所有 /api 请求的鉴权拦截器，未授权返回 401。"""
    if request.path.startswith("/api") and not _authorized():
        return jsonify({"error": "未授权：请提供 VF_WEB_TOKEN"}), 401
    return None


@app.errorhandler(Exception)
def _on_error(exc):
    """全局异常兜底，统一返回 JSON 错误信息。"""
    code = getattr(exc, "code", 500)
    return jsonify({"error": str(exc)[:500]}), code if isinstance(code, int) else 500


@app.errorhandler(FileNotFoundError)
def _on_missing(exc):
    """任务目录不存在时给 404 而不是 500：浏览器 localStorage 里常留着已删任务的 id。"""
    return jsonify({"error": str(exc)[:200] or "任务不存在"}), 404


@app.errorhandler(ValueError)
def _on_bad_request(exc):
    """参数不合法（非法 task_id、不支持的素材类型…）给 400，不是服务端故障。"""
    return jsonify({"error": str(exc)[:200] or "参数不合法"}), 400


# ---------------- 前端 ----------------
@app.get("/")
def index():
    """返回单页前端 index.html，禁用缓存保证前端改动即时生效。"""
    # 前端改了就要立刻生效，别让浏览器缓存旧 JS
    resp = send_from_directory(WEB_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _bg_video() -> dict:
    """挑一条成片给首页当动态背景：最近一个跑完且文件还在的任务。

    背景用自家产出的片子，不引任何外部素材——开发机是内网，外链图/视频根本拉不到。
    """
    for item in pipeline.list_tasks(20):
        tid = item.get("task_id")
        if item.get("status") != "completed" or not tid:
            continue
        try:
            final = (pipeline.load(tid).get("result") or {}).get("final") or ""
        except (OSError, ValueError):
            continue
        if final and os.path.isfile(os.path.join(pipeline.task_dir(tid), final)):
            return {"task_id": tid, "path": final}
    return {}


@app.get("/api/meta")
def meta():
    """前端启动时拉一次：默认参数、步骤名、是否需要 token、背景视频。"""
    return jsonify({"default_options": pipeline.DEFAULT_OPTIONS,
                    "steps": [{"key": k, "title": pipeline.STEP_TITLES[k],
                               "short": pipeline.STEP_SHORT.get(k, pipeline.STEP_TITLES[k])}
                              for k, _ in pipeline.STEPS],
                    "auth_required": bool(TOKEN),
                    "max_upload_mb": MAX_UPLOAD_MB,
                    "md5_dedupe": True,
                    "bg_video": _bg_video(),
                    "subtitle_font": pipeline._cjk_font()})


# ---------------- 任务 ----------------
@app.post("/api/tasks")
def create():
    """新建一个复刻任务并返回任务记录。"""
    body = request.get_json(silent=True) or {}
    rec = pipeline.create_task(body.get("product") or {}, body.get("options") or {},
                               title=body.get("title") or "")
    return jsonify(rec)


@app.get("/api/tasks")
def tasks():
    """返回全部任务列表。"""
    items = pipeline.list_tasks()
    # task.json 的 running 可能是服务重启前遗留的状态；下拉框只应把仍有
    # 后台线程的运行任务视为「运行中」，否则死任务会一直混在新建任务里。
    for item in items:
        thread = _running.get(item["task_id"])
        item["running"] = bool(thread and thread.is_alive())
    return jsonify(items)


def _fill_elapsed(rec: dict) -> None:
    """给正在跑的步骤补上 elapsed（秒）：进度条要显示「这一步已经跑了多久」。
    时间在服务端算，浏览器和开发机的时钟不一致也不会算歪。"""
    for st in (rec.get("steps") or {}).values():
        if st.get("status") == "running" and st.get("started_ts"):
            st["elapsed"] = round(time.time() - st["started_ts"], 1)

@app.get("/api/tasks/<task_id>")
def detail(task_id):
    """返回单个任务详情（状态/步骤/日志/结果）。"""
    rec = pipeline.load(task_id)
    rec["running"] = _is_running(task_id)
    _fill_elapsed(rec)
    return jsonify(rec)


def _is_running(task_id: str) -> bool:
    thread = _running.get(task_id)
    return bool(thread and thread.is_alive())


def _busy(task_id: str):
    """任务正在跑时拒掉改动类请求，返回 409 的响应；空闲返回 None。

    pipeline.run() 全程握着一份内存 rec，每步结束 save(rec) 是整份覆盖。这时候在
    HTTP 侧改参数 / 补素材 / 重置步骤，改动会在下一步结束时被那份旧 rec 原样写回去：
    改完几秒又变回来（用户看着像没保存），重置更糟——reset_from 已经把 generated/、
    render/ 挪进 history/ 了，旧 rec 又把 steps 的 done 状态写回来，task.json 声称
    各步完成、产物却已经不在原地。宁可让用户先终止任务再改。
    """
    if not _is_running(task_id):
        return None
    return jsonify({"error": "任务正在运行，先终止或等它跑完再改"}), 409


@app.patch("/api/tasks/<task_id>")
def patch(task_id):
    """修改任务的商品信息、复刻参数与任务名。"""
    busy = _busy(task_id)
    if busy:
        return busy
    body = request.get_json(silent=True) or {}
    return jsonify(pipeline.update_task(task_id, body.get("product"), body.get("options"),
                                        body.get("title")))


@app.delete("/api/tasks/<task_id>")
def drop(task_id):
    """删除任务，连同它的素材、中间结果和成片。正在跑的任务先终止再删。

    Python 线程杀不掉，所以先打终止标记：跑着的线程在下一次落盘时抛 TaskCancelled
    退出。必须等它真的不再写盘才能删目录，否则 _p() 的 makedirs 会把目录重建出来，
    任务看着像删不掉。
    """
    thread = _running.get(task_id)
    running = bool(thread and thread.is_alive())
    if running:
        pipeline.cancel_task(task_id)
        thread.join(timeout=8)
        print("[%s] 收到删除请求，已终止运行中的任务（线程%s）"
              % (task_id, "已退出" if not thread.is_alive() else "仍在收尾"), flush=True)
    alive = bool(thread and thread.is_alive())
    try:
        pipeline.delete_task(task_id)
    finally:
        # 标记留着会让同名任务永远存不了盘（task_id 带时间戳+随机串，撞不上），
        # 所以线程确实退出了才收。join 超时说明它还在跑长步骤（seedance 出片、ffmpeg
        # 合成期间几分钟不落盘），这时候收掉标记 = 放它继续跑，它下一次 save() 会顺着
        # _p() 的 makedirs 把刚 rmtree 的目录连 task.json 一起重建，已删的任务又回到列表里。
        # 留着标记，它下一次落盘时照旧抛 TaskCancelled 退出；/run 重跑前会主动 uncancel。
        if not alive:
            pipeline.uncancel_task(task_id)
            _running.pop(task_id, None)
    return jsonify({"deleted": task_id, "cancelled": running,
                    "thread_alive": alive})


def _safe_name(name: str) -> str:
    """清洗上传文件名，去掉路径分隔符与非法字符，防目录穿越。"""
    name = os.path.basename(name or "upload.bin").replace("\x00", "")
    return "".join(c for c in name if c not in '/\\:*?"<>|') or "upload.bin"


def _unique_upload_path(task_id: str, kind: str, name: str) -> str:
    """返回不会覆盖同名素材的上传路径。"""
    stem, ext = os.path.splitext(_safe_name(name))
    dst = pipeline._p(task_id, "assets", "uploads", kind, stem + ext)
    index = 2
    while os.path.exists(dst):
        dst = pipeline._p(task_id, "assets", "uploads", kind, "%s_%d%s" % (stem, index, ext))
        index += 1
    return dst


@app.get("/api/library/video-by-md5")
def video_by_md5():
    """上传前按 MD5 查询已有视频；命中后前端直接复用，避免再次传输大文件。"""
    digest = request.args.get("md5", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", digest):
        return jsonify({"error": "md5 必须是 32 位十六进制字符串"}), 400
    try:
        size = max(0, int(request.args.get("size", "0")))
    except ValueError:
        return jsonify({"error": "size 必须是整数"}), 400
    path = pipeline.find_reusable_video(digest, size)
    return jsonify({"found": bool(path), "path": path})


@app.get("/api/library/file-by-md5")
def file_by_md5():
    """上传前按 MD5 查询已有图片或视频。"""
    digest = request.args.get("md5", "").lower()
    media = request.args.get("media", "")
    if not re.fullmatch(r"[0-9a-f]{32}", digest):
        return jsonify({"error": "md5 必须是 32 位十六进制字符串"}), 400
    if media not in ("image", "video"):
        return jsonify({"error": "media 必须是 image 或 video"}), 400
    try:
        size = max(0, int(request.args.get("size", "0")))
    except ValueError:
        return jsonify({"error": "size 必须是整数"}), 400
    path = pipeline.find_reusable_file(digest, size, media)
    return jsonify({"found": bool(path), "path": path})


@app.post("/api/tasks/<task_id>/upload")
def upload(task_id):
    """上传某类素材（multipart，字段 kind + file）并登记到任务。"""
    busy = _busy(task_id)
    if busy:
        return busy
    # 先确认任务真的存在：_unique_upload_path 里的 _p() 会 makedirs、item.save() 会落盘，
    # 而整条链路上第一个校验任务存在的是最后那步 register_input 的 load()。
    # 前端 localStorage 里常留着已删任务的 id，那时接口虽然返回 404，磁盘上却已经多出
    # 一个 tasks/<不存在的id>/assets/uploads/，还会被素材库的 glob 当成「历史任务」扫出来。
    pipeline.load(task_id)
    kind = request.form.get("kind", "")
    if kind not in pipeline.KIND_EXTS:
        return jsonify({"error": "kind 必须是 %s" % "/".join(pipeline.KIND_EXTS)}), 400
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "没有收到文件"}), 400
    saved = []
    for item in files:
        # 按素材类型归档，避免参考片、商品图、人物图等全部混在同一个目录。
        dst = _unique_upload_path(task_id, kind, item.filename)
        item.save(dst)
        reusable = ""
        if kind in ("reference_video", "user_videos", "product_images"):
            digest = pipeline.file_md5(dst)
            claimed = request.form.get("md5", "").lower()
            if re.fullmatch(r"[0-9a-f]{32}", claimed) and digest != claimed:
                os.unlink(dst)
                return jsonify({"error": "素材 MD5 校验失败，请重新选择文件"}), 400
            media = "image" if kind == "product_images" else "video"
            reusable = pipeline.find_reusable_file(
                digest, os.path.getsize(dst), media, exclude=dst)
        if reusable:
            # 去重命中：把命中的文件硬链接到本任务的上传目录，别让 inputs 指向别的任务
            # 目录（那个任务一删，这条任务的素材就凭空消失）。硬链接不额外占磁盘。
            dst = pipeline.adopt_reusable(reusable, dst)
        pipeline.register_input(task_id, kind, dst)
        saved.append(os.path.basename(dst))
    rec = pipeline.load(task_id)
    return jsonify({"saved": saved, "inputs": rec["inputs"]})


@app.post("/api/tasks/<task_id>/remove")
def remove(task_id):
    """从任务里移除某个已上传的素材。"""
    busy = _busy(task_id)
    if busy:
        return busy
    body = request.get_json(silent=True) or {}
    rec = pipeline.remove_input(task_id, body.get("kind", ""), body.get("path", ""))
    return jsonify({"inputs": rec["inputs"]})


@app.get("/api/library")
def library():
    """可复用素材清单：按文件夹（分类）分组，含 media 类型供前端过滤。"""
    return jsonify(pipeline.list_library())


@app.get("/api/library/hot-videos")
def hot_videos():
    """单独返回爆款视频库，确保历史上传的参考视频在页面中始终可见。"""
    return jsonify({"files": pipeline.list_library().get("hot_videos", [])})


@app.get("/api/library/file")
def library_file():
    """预览素材库/历史任务里的单个文件。路径必须落在允许的根目录下，否则 404。

    带 download=1 时按附件返回，用来下载素材打包 zip。
    """
    path = os.path.realpath(request.args.get("path", ""))
    if not pipeline.is_reusable(path) or not os.path.isfile(path):
        return jsonify({"error": "文件不存在或路径越界"}), 404
    return send_file(path, conditional=True,
                     as_attachment=request.args.get("download") == "1")


@app.get("/api/library/pack/<name>")
def library_pack(name):
    """下载素材打包 zip。只认 output/library/downloads 下的文件名，避免拼绝对路径。"""
    safe = os.path.basename(name)
    root = os.path.realpath(os.path.join(pipeline.LIBRARY_DIR, "downloads"))
    path = os.path.realpath(os.path.join(root, safe))
    if not path.startswith(root + os.sep) or not os.path.isfile(path):
        return jsonify({"error": "素材包不存在"}), 404
    return send_file(path, as_attachment=True, conditional=True)


@app.get("/api/library/poster")
def library_poster():
    """视频首帧图。抽一次缓存到 output/library/.posters，列表就不必加载整段视频。"""
    path = os.path.realpath(request.args.get("path", ""))
    if not pipeline.is_reusable(path) or not os.path.isfile(path):
        return jsonify({"error": "文件不存在或路径越界"}), 404
    poster = os.path.join(POSTER_DIR, "%s.jpg" % pipeline.file_md5(path))
    if not os.path.isfile(poster):
        os.makedirs(POSTER_DIR, exist_ok=True)
        try:
            # 0.5s 而不是 0：片头常有一两帧纯黑，抽出来等于没有预览
            media.extract_frame(path, 0.5, poster, width=480)
        except (RuntimeError, OSError) as exc:
            return jsonify({"error": "抽帧失败：%s" % str(exc)[:120]}), 500
    return send_file(poster, conditional=True, max_age=86400)


@app.post("/api/tasks/<task_id>/pick")
def pick(task_id):
    """复用已有素材（body: kind + path），不重新上传。"""
    busy = _busy(task_id)
    if busy:
        return busy
    body = request.get_json(silent=True) or {}
    if body.get("kind") not in pipeline.KIND_EXTS:
        return jsonify({"error": "kind 必须是 %s" % "/".join(pipeline.KIND_EXTS)}), 400
    rec = pipeline.pick_input(task_id, body["kind"], body.get("path", ""))
    return jsonify({"inputs": rec["inputs"]})


@app.post("/api/tasks/<task_id>/run")
def run(task_id):
    """后台线程启动整条复刻流水线，已运行则返回 409。"""
    # 登记与判定必须在同一把锁里：两个标签页同时点「开始」的话，check-then-act 会放进
    # 两条流水线并行写同一个任务目录，_running 里还只剩后一个线程、前一个从此追不到。
    with _RUN_LOCK:
        if _is_running(task_id):
            return jsonify({"error": "该任务正在运行"}), 409
        pipeline.uncancel_task(task_id)      # 之前终止过的任务要能重新跑起来
        thread = threading.Thread(target=pipeline.run, args=(task_id,), daemon=True)
        _running[task_id] = thread
        thread.start()
    return jsonify({"started": True, "task_id": task_id})


@app.post("/api/tasks/<task_id>/reset")
def reset(task_id):
    """把任务重置到指定步骤以便重跑。"""
    busy = _busy(task_id)
    if busy:
        return busy
    body = request.get_json(silent=True) or {}
    return jsonify(pipeline.reset_from(task_id, body.get("step", "")))


def _read_json(path: str):
    """读任务目录里的中间产物；没跑到那步或写坏了都返回 None，前端据此显示占位。"""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


@app.get("/api/tasks/<task_id>/overview")
def overview(task_id):
    """关键信息视图：参考片拆解、商品事实卡、剧本、分段片时间线、成片。

    只挑前端要展示的那几份中间产物，免得用户在几十个文件里自己找。
    """
    root = os.path.realpath(pipeline.task_dir(task_id))
    if not os.path.isdir(root):
        raise FileNotFoundError("任务不存在：%s" % task_id)
    out = {key: _read_json(os.path.join(root, rel)) for key, rel in (
        ("reference", "reference/analysis.json"), ("product", "product/fact_card.json"),
        ("script", "script/script.json"), ("matches", "edit/asset_matches.json"),
        ("audio", "audio/reference_audio.json"),
        # 分镜级选图：事实卡只说哪些图当了全局参考图，真正进生成的是这里被镜头引用的那些
        ("shot_refs", "generated/shot_refs.json"))}
    segments = []
    for seg in _read_json(os.path.join(root, "generated", "segments.json")) or []:
        full = os.path.realpath(seg.get("file") or "")
        inside = full.startswith(root + os.sep)
        segments.append({"段号": seg.get("段号"), "mode": seg.get("mode") or "",
                         "时长秒": seg.get("生成时长秒"), "error": seg.get("error") or "",
                         "用户素材镜头": seg.get("用户素材镜头") or [],
                         "AI补片镜头": seg.get("AI补片镜头") or [],
                         "path": os.path.relpath(full, root) if inside else ""})
    out["segments"] = sorted(segments, key=lambda s: s.get("段号") or 0)
    rec = pipeline.load(task_id)
    out["final"] = (rec.get("result") or {}).get("final", "")
    return jsonify(out)


@app.get("/api/tasks/<task_id>/files")
def files(task_id):
    """列出任务目录下的全部产物，供前端随时翻中间结果——不管任务是跑完、跑挂还是跑一半。"""
    root = os.path.realpath(pipeline.task_dir(task_id))
    if not os.path.isdir(root):
        return jsonify({"error": "任务不存在"}), 404
    out = []
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            full = os.path.join(dirpath, name)
            if name.endswith(".tmp") or not os.path.isfile(full):
                continue
            out.append({"path": os.path.relpath(full, root),
                        "size_mb": round(os.path.getsize(full) / (1 << 20), 3)})
    out.sort(key=lambda item: item["path"])
    return jsonify({"files": out})


@app.get("/api/tasks/<task_id>/file")
def artifact(task_id):
    """取任务目录内的文件；路径必须落在任务目录里，防止越权读盘。

    带 download=1 时强制按附件下载：成片默认是内联播放，点「下载」要的是存到本地。
    """
    root = os.path.realpath(pipeline.task_dir(task_id))
    target = os.path.realpath(os.path.join(root, request.args.get("path", "")))
    if not (target == root or target.startswith(root + os.sep)) or not os.path.isfile(target):
        return jsonify({"error": "文件不存在或路径越界"}), 404
    inline = os.path.splitext(target)[1].lower() in (".mp4", ".jpg", ".jpeg", ".png", ".webp",
                                                    ".md", ".json", ".srt", ".txt")
    force = request.args.get("download") == "1"
    return send_file(target, as_attachment=force or not inline, conditional=True)


def main():
    """命令行入口：读取环境变量并启动 Flask 服务。"""
    # ==== 服务参数：改这里或用环境变量覆盖 ====
    host = os.getenv("VF_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("VF_WEB_PORT", "8420"))
    # =========================================
    os.makedirs(pipeline.TASKS_DIR, exist_ok=True)
    # 素材库根目录先建出来。分类子目录由用户自己按题材建（小米手机/汽车/…），
    # 一个目录里图片视频混放都行，前端按当前素材位过滤。
    os.makedirs(pipeline.LIBRARY_DIR, exist_ok=True)
    if not TOKEN:
        print("⚠ 未设置 VF_WEB_TOKEN：接口无鉴权，任何能访问 %s:%d 的人都能建任务并读产物。"
              % (host, port))
        print("  建议：export VF_WEB_TOKEN=<自定义口令> 后再启动，或只用 SSH 端口转发访问。")
    print("ViralForge Web 已启动： http://%s:%d" % ("127.0.0.1" if host == "0.0.0.0" else host,
                                                    port))
    print("  任务产物目录：%s" % pipeline.TASKS_DIR)
    print("  可复用素材库：%s" % pipeline.LIBRARY_DIR)
    app.run(host=host, port=port, threaded=True, debug=False,
            request_handler=TokenSafeHandler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
