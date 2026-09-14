#!/usr/bin/env bash
# 重新渲染新版汇报图、可编辑 Excalidraw 源，以及保留的旧版 Graphviz 图。
# 依赖：python3 + Pillow；Excalidraw 额外依赖 gen-graph skill；旧图依赖 dot。
set -e
cd "$(dirname "$0")"

python3 src/render_enterprise_figures.py

graph_scripts="${GEN_GRAPH_SCRIPTS:-/root/.comate/skills/gen-graph/scripts}"
if [ -f "$graph_scripts/layout_engine.py" ]; then
  for graph in src/vf*.graph.json; do
    layout="${graph%.graph.json}.layout.json"
    source="${graph%.graph.json}.excalidraw"
    python3 "$graph_scripts/layout_engine.py" "$graph" "$layout"
    python3 "$graph_scripts/validate_layout.py" "$layout"
    python3 "$graph_scripts/excalidraw_builder.py" "$layout" "$source"
  done
else
  echo "skipped Excalidraw: gen-graph scripts not found at $graph_scripts"
fi

for f in src/*.dot; do
  out="$(basename "${f%.dot}").png"
  dot -Tpng "$f" -o "$out"
  echo "rendered legacy: $out"
done
