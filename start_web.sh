#!/usr/bin/env bash
# ViralForge Web 服务启动脚本：需在开发机的真实 shell（SSH 会话）里执行，
# 这样监听才落在宿主机网卡上，内网其他机器才访问得到。
#
# 用法：
#   ./start_web.sh            前台启动（Ctrl+C 停止）
#   ./start_web.sh -d         后台常驻（日志 /tmp/viralforge_web.log）
#   ./start_web.sh stop       停止后台服务
# 环境变量：VF_WEB_PORT（默认 8420）、VF_WEB_TOKEN（访问口令，强烈建议设置）
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PORT="${VF_WEB_PORT:-8420}"
LOG=/tmp/viralforge_web.log
PID=/tmp/viralforge_web.pid

if [ "${1:-}" = "stop" ]; then
  [ -f "$PID" ] && kill "$(cat "$PID")" 2>/dev/null && rm -f "$PID" && echo "已停止" || echo "没有在运行的后台服务"
  exit 0
fi

# 端口被自己的旧进程占着就先收掉：对外链接必须每次都是同一个，不能因为占用而换端口
if command -v ss >/dev/null && ss -ltnp 2>/dev/null | grep -q ":${PORT} "; then
  if pgrep -f "python3 server.py" >/dev/null 2>&1; then
    echo "端口 ${PORT} 被旧的服务占着，先停掉它"
    pkill -f "python3 server.py" || true
    sleep 2
  fi
fi
if command -v ss >/dev/null && ss -ltnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 被别的程序占用，链接会变，请先处理："
  ss -ltnp 2>/dev/null | grep ":${PORT} "
  exit 1
fi

if [ -z "${VF_WEB_TOKEN:-}" ]; then
  VF_WEB_TOKEN=$(sed -n 's/^VF_WEB_TOKEN=//p' config.env 2>/dev/null | head -1)
fi
if [ -z "${VF_WEB_TOKEN:-}" ]; then
  echo "⚠ 未设置 VF_WEB_TOKEN：内网任何人都能建任务、消耗模型额度、下载产物。"
  echo "  建议：在 config.env 里写 VF_WEB_TOKEN=<口令>，或 export VF_WEB_TOKEN=..."
fi

IP=$(hostname -i 2>/dev/null | awk '{print $1}')
if [ -n "${VF_WEB_TOKEN:-}" ]; then
  echo "打开这个地址即可（口令已带在链接里）："
  echo "  http://${IP}:${PORT}/?token=${VF_WEB_TOKEN}"
else
  echo "内网访问地址： http://${IP}:${PORT}   （本机 http://127.0.0.1:${PORT}）"
fi

export VF_WEB_PORT="$PORT"
if [ "${1:-}" = "-d" ]; then
  nohup python3 server.py >"$LOG" 2>&1 &
  echo $! >"$PID"
  sleep 2
  CODE=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' \
         "http://127.0.0.1:${PORT}/api/meta?token=${VF_WEB_TOKEN:-}" || true)
  if [ "$CODE" = "200" ]; then
    echo "已后台启动，pid $(cat "$PID")，日志 $LOG"
  else
    echo "启动异常（自检 HTTP ${CODE}），日志尾部："; tail -20 "$LOG"; exit 1
  fi
else
  exec python3 server.py
fi
