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
  PATCH/api/tasks/<id>                改商品信息与复刻参数
  POST /api/tasks/<id>/upload         上传素材（multipart，字段 kind + file）
  POST /api/tasks/<id>/run            后台跑流水线
  POST /api/tasks/<id>/reset          从某步骤重跑
  GET  /api/tasks/<id>/file?path=     取任务目录内的产物（成片、md、json）
"""
import os
import threading

from flask import Flask, jsonify, request, send_file, send_from_directory

import pipeline  # pyright: ignore[reportImplicitRelativeImport]

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
TOKEN = os.getenv("VF_WEB_TOKEN", "").strip()
MAX_UPLOAD_MB = int(os.getenv("VF_MAX_UPLOAD_MB", "512"))

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * (1 << 20)

_running: "dict[str, threading.Thread]" = {}


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


# ---------------- 前端 ----------------
@app.get("/")
def index():
    """返回单页前端 index.html，禁用缓存保证前端改动即时生效。"""
    # 前端改了就要立刻生效，别让浏览器缓存旧 JS
    resp = send_from_directory(WEB_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/meta")
def meta():
    """前端启动时拉一次：默认参数、步骤名、是否需要 token。"""
    return jsonify({"default_options": pipeline.DEFAULT_OPTIONS,
                    "steps": [{"key": k, "title": pipeline.STEP_TITLES[k]}
                              for k, _ in pipeline.STEPS],
                    "auth_required": bool(TOKEN),
                    "max_upload_mb": MAX_UPLOAD_MB,
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
    return jsonify(pipeline.list_tasks())


@app.get("/api/tasks/<task_id>")
def detail(task_id):
    """返回单个任务详情（状态/步骤/日志/结果）。"""
    rec = pipeline.load(task_id)
    rec["running"] = task_id in _running and _running[task_id].is_alive()
    return jsonify(rec)


@app.patch("/api/tasks/<task_id>")
def patch(task_id):
    """修改任务的商品信息与复刻参数。"""
    body = request.get_json(silent=True) or {}
    return jsonify(pipeline.update_task(task_id, body.get("product"), body.get("options"),
                                        body.get("title")))


def _safe_name(name: str) -> str:
    """清洗上传文件名，去掉路径分隔符与非法字符，防目录穿越。"""
    name = os.path.basename(name or "upload.bin").replace("\x00", "")
    return "".join(c for c in name if c not in '/\\:*?"<>|') or "upload.bin"


@app.post("/api/tasks/<task_id>/upload")
def upload(task_id):
    """上传某类素材（multipart，字段 kind + file）并登记到任务。"""
    kind = request.form.get("kind", "")
    if kind not in pipeline.KIND_EXTS:
        return jsonify({"error": "kind 必须是 %s" % "/".join(pipeline.KIND_EXTS)}), 400
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "没有收到文件"}), 400
    saved = []
    for item in files:
        dst = pipeline._p(task_id, "assets", "uploads", _safe_name(item.filename))
        item.save(dst)
        pipeline.register_input(task_id, kind, dst)
        saved.append(os.path.basename(dst))
    rec = pipeline.load(task_id)
    return jsonify({"saved": saved, "inputs": rec["inputs"]})


@app.post("/api/tasks/<task_id>/remove")
def remove(task_id):
    """从任务里移除某个已上传的素材。"""
    body = request.get_json(silent=True) or {}
    rec = pipeline.remove_input(task_id, body.get("kind", ""), body.get("path", ""))
    return jsonify({"inputs": rec["inputs"]})


@app.post("/api/tasks/<task_id>/run")
def run(task_id):
    """后台线程启动整条复刻流水线，已运行则返回 409。"""
    if task_id in _running and _running[task_id].is_alive():
        return jsonify({"error": "该任务正在运行"}), 409
    thread = threading.Thread(target=pipeline.run, args=(task_id,), daemon=True)
    _running[task_id] = thread
    thread.start()
    return jsonify({"started": True, "task_id": task_id})


@app.post("/api/tasks/<task_id>/reset")
def reset(task_id):
    """把任务重置到指定步骤以便重跑。"""
    body = request.get_json(silent=True) or {}
    return jsonify(pipeline.reset_from(task_id, body.get("step", "")))


@app.get("/api/tasks/<task_id>/file")
def artifact(task_id):
    """取任务目录内的文件；路径必须落在任务目录里，防止越权读盘。"""
    root = os.path.realpath(pipeline.task_dir(task_id))
    target = os.path.realpath(os.path.join(root, request.args.get("path", "")))
    if not (target == root or target.startswith(root + os.sep)) or not os.path.isfile(target):
        return jsonify({"error": "文件不存在或路径越界"}), 404
    inline = os.path.splitext(target)[1].lower() in (".mp4", ".jpg", ".jpeg", ".png", ".webp",
                                                    ".md", ".json", ".srt", ".txt")
    return send_file(target, as_attachment=not inline, conditional=True)


def main():
    """命令行入口：读取环境变量并启动 Flask 服务。"""
    # ==== 服务参数：改这里或用环境变量覆盖 ====
    host = os.getenv("VF_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("VF_WEB_PORT", "8420"))
    # =========================================
    os.makedirs(pipeline.TASKS_DIR, exist_ok=True)
    if not TOKEN:
        print("⚠ 未设置 VF_WEB_TOKEN：接口无鉴权，任何能访问 %s:%d 的人都能建任务并读产物。"
              % (host, port))
        print("  建议：export VF_WEB_TOKEN=<自定义口令> 后再启动，或只用 SSH 端口转发访问。")
    print("ViralForge Web 已启动： http://%s:%d" % ("127.0.0.1" if host == "0.0.0.0" else host,
                                                    port))
    app.run(host=host, port=port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

