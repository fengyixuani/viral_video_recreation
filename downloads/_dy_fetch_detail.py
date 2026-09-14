# -*- coding: utf-8 -*-
"""临时脚本：解析抖音视频详情（a_bogus 签名），算法取自 JoeanAmier/Douyin_TikTok_Download_API。"""
import json
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, "/tmp/dydl")
sys.path.insert(0, "/tmp/dylib")
from abogus import ABogus  # noqa: E402

AWEME_ID = sys.argv[1] if len(sys.argv) > 1 else "7683845933591296635"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"

params = {
    "device_platform": "webapp",
    "aid": "6383",
    "channel": "channel_pc_web",
    "pc_client_type": "1",
    "version_code": "290100",
    "version_name": "29.1.0",
    "cookie_enabled": "true",
    "screen_width": "1920",
    "screen_height": "1080",
    "browser_language": "zh-CN",
    "browser_platform": "Win32",
    "browser_name": "Chrome",
    "browser_version": "130.0.0.0",
    "browser_online": "true",
    "engine_name": "Blink",
    "engine_version": "130.0.0.0",
    "os_name": "Windows",
    "os_version": "10",
    "cpu_core_num": "12",
    "device_memory": "8",
    "platform": "PC",
    "downlink": "10",
    "effective_type": "4g",
    "from_user_page": "1",
    "locate_query": "false",
    "need_time_list": "1",
    "pc_libra_divert": "Windows",
    "publish_video_strategy_type": "2",
    "round_trip_time": "0",
    "show_live_replay_strategy": "1",
    "time_list_query": "0",
    "whale_cut_token": "",
    "update_version_code": "170400",
    "aweme_id": AWEME_ID,
    "msToken": "",
}

a_bogus = ABogus().get_value(params)
url = ("https://www.douyin.com/aweme/v1/web/aweme/detail/?" + urllib.parse.urlencode(params)
       + "&a_bogus=" + urllib.parse.quote(a_bogus, safe=""))

req = urllib.request.Request(url, headers={
    "User-Agent": UA,
    "Referer": "https://www.douyin.com/",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
    "Cookie": "ttwid=1%7Ca0hwcz35h3agD4tMMIQ3Nh7m55Q5svzLCOSd-xwWJ_8%7C1789047086%7C6ce3f59324a00293478997c53211a3088f79917e49f8380396edbd3debc9c598",
})
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))

detail = data.get("aweme_detail") or {}
print("status_code:", data.get("status_code"))
print("desc:", (detail.get("desc") or "")[:80])
video = detail.get("video") or {}
play = (video.get("play_addr") or {}).get("url_list") or []
print("play urls:", len(play))
for u in play:
    print("  ", u[:160])
with open("/tmp/dydl/detail.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
