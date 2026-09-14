"""临时自检：前端任务落到 output/web/tasks、素材库跨两个根、上传接口正常。跑完即删。"""
import io
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["VF_WEB_TOKEN"] = "checktoken"

import server  # noqa: E402
import pipeline  # noqa: E402
import task_store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
assert pipeline.TASKS_DIR == os.path.join(HERE, "output", "web", "tasks"), pipeline.TASKS_DIR
assert task_store.CLI_TASKS_DIR == os.path.join(HERE, "output", "tasks")
print("前端任务根:", pipeline.TASKS_DIR)
print("素材扫描根:", task_store.TASK_ROOTS)

c = server.app.test_client()
H = {"Authorization": "Bearer checktoken"}

rec = c.post("/api/tasks", json={"product": {"name": "落盘自检"}}, headers=H).get_json()
tid = rec["task_id"]
assert os.path.isdir(os.path.join(pipeline.TASKS_DIR, tid)), "任务没落在 output/web/tasks"
assert not os.path.isdir(os.path.join(task_store.CLI_TASKS_DIR, tid)), "串到 output/tasks 了"
print("建任务落盘: output/web/tasks/%s" % tid)

r = c.post("/api/tasks/%s/upload" % tid, headers=H, content_type="multipart/form-data",
           data={"kind": "reference_video", "file": (io.BytesIO(b"\x00" * 2048), "自检片.mp4")})
body = r.get_json()
print("上传:", r.status_code, body.get("saved"))
assert r.status_code == 200 and body["saved"] == ["自检片.mp4"], body
saved = body["inputs"]["reference_video"]
assert saved.startswith(pipeline.TASKS_DIR), saved
print("上传落盘:", os.path.relpath(saved, HERE))

r = c.post("/api/tasks/%s/upload" % tid, headers=H, content_type="multipart/form-data",
           data={"kind": "reference_video"})
print("无 file 字段:", r.status_code, r.get_json().get("error"))
assert r.status_code == 400

lib = c.get("/api/library", headers=H).get_json()["library"]
names = [f["name"] for f in lib["reference_video"]]
assert "自检片.mp4" in names, names
print("素材库 reference_video:", names)

nxt = c.post("/api/tasks", json={}, headers=H).get_json()
r = c.post("/api/tasks/%s/pick" % nxt["task_id"],
           json={"kind": "reference_video", "path": saved}, headers=H)
assert r.status_code == 200 and r.get_json()["inputs"]["reference_video"] == saved, r.get_json()
print("跨任务复用 OK")

r = c.post("/api/tasks/%s/pick" % nxt["task_id"],
           json={"kind": "reference_video", "path": "/etc/hosts"}, headers=H)
assert r.status_code != 200
print("越界路径仍被拒:", r.get_json().get("error"))

r = c.get("/api/tasks/nosuchtask", headers=H)
print("不存在的任务:", r.status_code)
assert r.status_code == 404, r.status_code

for t in (tid, nxt["task_id"]):
    shutil.rmtree(os.path.join(pipeline.TASKS_DIR, t), ignore_errors=True)
print("OK（自检任务已清理）")
