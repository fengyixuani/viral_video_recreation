"""临时脚本：批量拉取候选 B 站短视频 + 抽帧做联系表，用于人工/模型筛选是否适合动作迁移。用完即删。"""
import json
import os
import subprocess
import sys
import urllib.request

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cands")
os.makedirs(OUT, exist_ok=True)


def get(url, referer="https://www.bilibili.com/"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
    return urllib.request.urlopen(req, timeout=60).read()


def fetch(bvid):
    view = json.loads(get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"))["data"]
    cid, title, dur = view["cid"], view["title"], view["duration"]
    play = json.loads(get(
        f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn=120&fnval=4048&fourk=1",
        referer=f"https://www.bilibili.com/video/{bvid}/"))["data"]["dash"]
    v = max([x for x in play["video"] if x["codecs"].startswith("avc1")],
            key=lambda x: (x["id"], x["bandwidth"]))
    a = max(play["audio"], key=lambda x: x["bandwidth"])
    vp, ap = f"{OUT}/{bvid}_v.m4s", f"{OUT}/{bvid}_a.m4s"
    for item, path in ((v, vp), (a, ap)):
        with open(path, "wb") as f:
            f.write(get(item["baseUrl"], referer=f"https://www.bilibili.com/video/{bvid}/"))
    mp4 = f"{OUT}/{bvid}.mp4"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", vp, "-i", ap,
                    "-c", "copy", mp4], check=True)
    os.remove(vp)
    os.remove(ap)
    # 3 帧联系表：20% / 50% / 80% 处
    sheet = f"{OUT}/{bvid}_sheet.jpg"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", mp4,
                    "-vf", f"select='eq(n\\,{0})+gte(t\\,{dur*0.2})*lte(t\\,{dur*0.2+0.05})',"
                           "scale=-1:360", "-frames:v", "1", f"{OUT}/{bvid}_f1.jpg"], check=True)
    for i, t in enumerate([dur * 0.2, dur * 0.5, dur * 0.8], start=1):
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}",
                        "-i", mp4, "-frames:v", "1", "-vf", "scale=-2:360",
                        f"{OUT}/{bvid}_f{i}.jpg"], check=True)
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", f"{OUT}/{bvid}_f1.jpg", "-i", f"{OUT}/{bvid}_f2.jpg",
                    "-i", f"{OUT}/{bvid}_f3.jpg", "-filter_complex",
                    "[0][1][2]hstack=inputs=3", sheet], check=True)
    for i in (1, 2, 3):
        os.remove(f"{OUT}/{bvid}_f{i}.jpg")
    probe = subprocess.run([FF, "-hide_banner", "-i", mp4], capture_output=True, text=True).stderr
    res = [l.strip() for l in probe.splitlines() if "Video:" in l]
    return {"bvid": bvid, "title": title, "duration": dur, "mp4": mp4, "sheet": sheet,
            "stream": res[0] if res else ""}


if __name__ == "__main__":
    for bvid in sys.argv[1:]:
        try:
            info = fetch(bvid)
            print(json.dumps(info, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"bvid": bvid, "error": repr(exc)}, ensure_ascii=False))
