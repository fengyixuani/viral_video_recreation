#!/usr/bin/env python3
"""Render the business-facing ViralForge diagrams.

The polished SVG/PNG files use a fixed 16:9 presentation layout so that the
figures remain orderly in Markdown, slides, and exported documents.  The same
content is also emitted as gen-graph intermediate JSON; render.sh converts
those files into editable Excalidraw sources.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1920
HEIGHT = 1080
FIGURES_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(__file__).resolve().parent
FONT_FILE = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")

FONT = "ViralForgeCJK, 'Droid Sans Fallback', sans-serif"
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

NAVY = "#0B2A5B"
BLUE = "#1F5AA6"
BLUE_2 = "#3778C2"
PALE_BLUE = "#EAF1FB"
PALE_BLUE_2 = "#F3F7FC"
ORANGE = "#F28C28"
ORANGE_DARK = "#D86F0B"
PALE_ORANGE = "#FFF1E3"
INK = "#17324D"
MUTED = "#60758C"
LINE = "#D7E1ED"
SOFT = "#F7F9FC"
WHITE = "#FFFFFF"
GREEN = "#278A67"
PALE_GREEN = "#E8F6F0"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _raster_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(str(FONT_FILE), size)
    return _FONT_CACHE[size]


class Svg:
    def __init__(self, title: str):
        self.title = title
        self.image = Image.new("RGBA", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.parts = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
            ),
            "<defs>",
            (
                "<style>"
                "@font-face {"
                "font-family: ViralForgeCJK;"
                "src: url('file:///usr/share/fonts/google-droid/DroidSansFallback.ttf');"
                "font-style: normal;"
                "font-weight: 100 900;"
                "}"
                "</style>"
            ),
            (
                '<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">'
                '<feDropShadow dx="0" dy="5" stdDeviation="8" '
                'flood-color="#17324D" flood-opacity="0.10"/>'
                "</filter>"
            ),
            (
                '<marker id="arrow-blue" markerWidth="12" markerHeight="12" '
                'refX="10" refY="6" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M0,0 L12,6 L0,12 z" fill="{BLUE}"/>'
                "</marker>"
            ),
            (
                '<marker id="arrow-orange" markerWidth="12" markerHeight="12" '
                'refX="10" refY="6" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M0,0 L12,6 L0,12 z" fill="{ORANGE}"/>'
                "</marker>"
            ),
            "</defs>",
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{WHITE}"/>',
        ]

    def _shadow(self, draw_shape) -> None:
        layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(layer)
        draw_shape(shadow_draw)
        layer = layer.filter(ImageFilter.GaussianBlur(8))
        self.image.alpha_composite(layer)
        self.draw = ImageDraw.Draw(self.image)

    def _raster_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
        width: int,
        dashed: bool,
    ) -> None:
        if not dashed:
            self.draw.line([start, end], fill=color, width=width)
            return
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        cursor = 0.0
        while cursor < length:
            stop = min(cursor + 8.0, length)
            self.draw.line(
                [
                    (x1 + ux * cursor, y1 + uy * cursor),
                    (x1 + ux * stop, y1 + uy * stop),
                ],
                fill=color,
                width=width,
            )
            cursor += 16.0

    def _raster_arrowhead(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
        width: int,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 9 + width * 2
        half = 4 + width
        base_x = x2 - math.cos(angle) * length
        base_y = y2 - math.sin(angle) * length
        perp_x = -math.sin(angle) * half
        perp_y = math.cos(angle) * half
        self.draw.polygon(
            [
                (x2, y2),
                (base_x + perp_x, base_y + perp_y),
                (base_x - perp_x, base_y - perp_y),
            ],
            fill=color,
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = WHITE,
        stroke: str = LINE,
        sw: float = 1.4,
        radius: float = 16,
        shadow: bool = False,
        opacity: float = 1.0,
    ) -> None:
        shadow_attr = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
            f'stroke-width="{sw}"{shadow_attr}/>'
        )
        if shadow:
            self._shadow(
                lambda draw: draw.rounded_rectangle(
                    (x, y + 4, x + w, y + h + 4),
                    radius=radius,
                    fill=(23, 50, 77, 22),
                )
            )
        self.draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=radius,
            fill=fill,
            outline=None if stroke == "none" else stroke,
            width=max(1, round(sw)),
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = WHITE,
        stroke: str = LINE,
        sw: float = 1.5,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )
        self.draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=fill,
            outline=None if stroke == "none" else stroke,
            width=max(1, round(sw)),
        )

    def diamond(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        *,
        fill: str = PALE_ORANGE,
        stroke: str = ORANGE,
        sw: float = 2,
        shadow: bool = False,
    ) -> None:
        points = (
            f"{cx},{cy - h / 2} {cx + w / 2},{cy} "
            f"{cx},{cy + h / 2} {cx - w / 2},{cy}"
        )
        shadow_attr = ' filter="url(#shadow)"' if shadow else ""
        self.parts.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{shadow_attr}/>'
        )
        raster_points = [
            (cx, cy - h / 2),
            (cx + w / 2, cy),
            (cx, cy + h / 2),
            (cx - w / 2, cy),
        ]
        self.draw.polygon(raster_points, fill=fill)
        self.draw.line(
            raster_points + [raster_points[0]],
            fill=stroke,
            width=max(1, round(sw)),
            joint="curve",
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = LINE,
        sw: float = 2,
        dashed: bool = False,
        arrow: bool = False,
        arrow_color: str = "blue",
    ) -> None:
        dash_attr = ' stroke-dasharray="8 8"' if dashed else ""
        arrow_attr = (
            f' marker-end="url(#arrow-{arrow_color})"' if arrow else ""
        )
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"'
            f"{dash_attr}{arrow_attr}/>"
        )
        raster_width = max(1, round(sw))
        self._raster_line(
            (x1, y1),
            (x2, y2),
            color=color,
            width=raster_width,
            dashed=dashed,
        )
        if arrow:
            marker_color = ORANGE if arrow_color == "orange" else BLUE
            self._raster_arrowhead(
                (x1, y1),
                (x2, y2),
                color=marker_color,
                width=raster_width,
            )

    def path(
        self,
        points: list[tuple[float, float]],
        *,
        color: str = BLUE,
        sw: float = 2,
        dashed: bool = False,
        arrow: bool = False,
        arrow_color: str = "blue",
    ) -> None:
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = ' stroke-dasharray="8 8"' if dashed else ""
        arrow_attr = (
            f' marker-end="url(#arrow-{arrow_color})"' if arrow else ""
        )
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round"{dash_attr}{arrow_attr}/>'
        )
        raster_width = max(1, round(sw))
        for start, end in zip(points, points[1:]):
            self._raster_line(
                start,
                end,
                color=color,
                width=raster_width,
                dashed=dashed,
            )
        if arrow and len(points) >= 2:
            marker_color = ORANGE if arrow_color == "orange" else BLUE
            self._raster_arrowhead(
                points[-2],
                points[-1],
                color=marker_color,
                width=raster_width,
            )

    def text(
        self,
        x: float,
        y: float,
        value: object,
        *,
        size: int = 18,
        color: str = INK,
        weight: int = 400,
        anchor: str = "start",
        baseline: str = "middle",
        letter_spacing: float = 0,
    ) -> None:
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}" letter-spacing="{letter_spacing}">'
            f"{_esc(value)}</text>"
        )
        raster_anchor = {
            "start": "lm",
            "middle": "mm",
            "end": "rm",
        }.get(anchor, "lm")
        synthetic_bold = 1 if weight >= 700 and size >= 18 else 0
        self.draw.text(
            (x, y),
            str(value),
            font=_raster_font(size),
            fill=color,
            anchor=raster_anchor,
            stroke_width=synthetic_bold,
            stroke_fill=color,
        )

    def lines(
        self,
        x: float,
        y: float,
        values: list[str] | tuple[str, ...],
        *,
        size: int = 17,
        color: str = MUTED,
        weight: int = 400,
        gap: int = 28,
        anchor: str = "start",
    ) -> None:
        for index, value in enumerate(values):
            self.text(
                x,
                y + index * gap,
                value,
                size=size,
                color=color,
                weight=weight,
                anchor=anchor,
            )

    def pill(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        fill: str = PALE_BLUE,
        color: str = BLUE,
        stroke: str | None = None,
        size: int = 15,
        weight: int = 600,
    ) -> None:
        self.rect(
            x,
            y,
            w,
            h,
            fill=fill,
            stroke=stroke or fill,
            radius=h / 2,
            sw=1,
        )
        self.text(
            x + w / 2,
            y + h / 2 + 1,
            label,
            size=size,
            color=color,
            weight=weight,
            anchor="middle",
        )

    def header(
        self,
        title: str,
        subtitle: str,
        *,
        tag: str | None = None,
    ) -> None:
        self.rect(70, 42, 5, 82, fill=ORANGE, stroke=ORANGE, radius=0)
        self.text(100, 68, title, size=36, color=NAVY, weight=700)
        self.text(100, 110, subtitle, size=18, color=MUTED, weight=400)
        self.line(70, 150, 1850, 150, color="#DDE3EB", sw=1)
        if tag:
            self.line(1385, 68, 1435, 68, color=ORANGE, sw=3)
            self.text(
                1850,
                68,
                tag,
                size=14,
                color=NAVY,
                weight=600,
                anchor="end",
                letter_spacing=0.4,
            )

    def section_label(
        self,
        y: float,
        h: float,
        number: str,
        title: str,
        subtitle: str,
    ) -> None:
        self.text(72, y + 28, number, size=15, color=ORANGE, weight=700)
        self.line(72, y + 50, 210, y + 50, color=LINE, sw=1)
        self.text(72, y + 82, title, size=22, color=NAVY, weight=700)
        self.lines(
            72,
            y + 116,
            subtitle.split("\n"),
            size=13,
            color=MUTED,
            gap=21,
        )

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: list[str] | tuple[str, ...],
        *,
        number: str | None = None,
        accent: str = BLUE,
        fill: str = WHITE,
        title_color: str = NAVY,
        body_color: str = MUTED,
        shadow: bool = True,
        body_size: int = 16,
        body_gap: int = 25,
        badge: str | None = None,
    ) -> None:
        self.rect(
            x,
            y,
            w,
            h,
            fill=fill,
            stroke=LINE,
            radius=6,
            sw=1,
            shadow=False,
        )
        self.rect(x + 24, y + 18, 44, 3, fill=accent, stroke=accent, radius=0)
        title_x = x + 24
        if number:
            self.text(
                x + 24,
                y + 43,
                number,
                size=13,
                color=accent,
                weight=700,
            )
            title_x = x + 58
        self.text(title_x, y + 43, title, size=20, color=title_color, weight=700)
        if badge:
            self.text(
                x + w - 24,
                y + 43,
                badge,
                size=12,
                color=accent,
                weight=700,
                anchor="end",
            )
        self.lines(
            x + 24,
            y + 78,
            body,
            size=body_size,
            color=body_color,
            gap=body_gap,
        )

    def footer(self, text: str, *, lead: str = "核心结论") -> None:
        self.line(70, 962, 1850, 962, color=LINE, sw=1)
        self.rect(70, 988, 5, 34, fill=ORANGE, stroke=ORANGE, radius=0)
        self.text(96, 1005, lead, size=13, color=ORANGE_DARK, weight=700)
        self.text(220, 1005, text, size=20, color=NAVY, weight=600)

    def render(self, stem: str) -> None:
        self.parts.append("</svg>")
        svg_text = "\n".join(self.parts)
        svg_path = FIGURES_DIR / f"{stem}.svg"
        png_path = FIGURES_DIR / f"{stem}.png"
        svg_path.write_text(svg_text, encoding="utf-8")
        self.image.convert("RGB").save(png_path, optimize=True)
        print(f"rendered: {png_path.name}")


def _agent_guardrails(c: Svg) -> None:
    """Right panel: what the Agent is allowed to decide, and who says no."""
    c.rect(1450, 272, 400, 470, fill=WHITE, stroke=LINE, radius=6, sw=1)
    c.text(1476, 302, "02 / GUARDRAILS", size=12, color=ORANGE, weight=700)
    c.text(1476, 338, "决策依据与护栏", size=22, color=NAVY, weight=700)
    c.text(1476, 366, "Agent 有自由度，但边界是写死的", size=13, color=MUTED)
    c.line(1476, 388, 1824, 388, color="#E3E9F1", sw=1)

    blocks = [
        (
            404,
            ORANGE,
            "rules 策略表",
            ["规定允许哪些动作、谁优先", "自上而下命中，末行必须兜底"],
        ),
        (
            520,
            GREEN,
            "gates 验收门",
            ["对真实产物实测，不看模型自评", "不通过就不许进入成片"],
        ),
        (
            636,
            NAVY,
            "安全与诚实约束",
            ["真人风控：改用线稿替身兜底", "显式失败，不静默丢内容"],
        ),
    ]
    for y, accent, title, items in blocks:
        c.rect(1476, y, 40, 3, fill=accent, stroke=accent, radius=0)
        c.text(1476, y + 26, title, size=19, color=NAVY, weight=700)
        c.lines(1476, y + 56, items, size=13, color=MUTED, gap=24)

    c.text(1425, 434, "查表", size=11, color=MUTED, weight=600, anchor="middle")
    c.line(1402, 452, 1448, 452, color=ORANGE, sw=2, arrow=True, arrow_color="orange")
    c.text(1425, 572, "返回动作", size=11, color=MUTED, weight=600, anchor="middle")
    c.line(1448, 590, 1402, 590, color=ORANGE, sw=2, arrow=True, arrow_color="orange")


def _agent_capabilities(c: Svg) -> None:
    """Bottom band: interchangeable tools that execute but never decide."""
    c.rect(70, 786, 1780, 120, fill=SOFT, stroke=LINE, radius=6, sw=1)
    c.text(96, 812, "03 / CAPABILITY POOL", size=12, color=ORANGE, weight=700)
    c.text(96, 844, "可插拔能力池", size=21, color=NAVY, weight=700)
    c.text(96, 876, "工具只负责执行，探活失败就换一条路", size=13, color=MUTED)

    columns = [
        (430, "理解能力", "Gemini / Qwen 读画面与声音"),
        (780, "生成能力", "Seedream 出图 / Seedance 出片"),
        (1130, "媒体能力", "ffmpeg / demucs / librosa"),
        (1480, "外挂能力", "音色克隆 / 字幕克隆 / ASR"),
    ]
    for x, name, detail in columns:
        c.line(x - 32, 812, x - 32, 882, color="#E3E9F1", sw=1)
        c.text(x, 838, name, size=18, color=INK, weight=700)
        c.text(x, 870, detail, size=13, color=MUTED)

    c.line(800, 748, 800, 782, color=ORANGE, sw=2, arrow=True, arrow_color="orange")
    c.text(818, 766, "调度", size=12, color=ORANGE_DARK, weight=600)
    c.line(1120, 782, 1120, 748, color=BLUE, sw=2, arrow=True)
    c.text(1138, 766, "产物回流", size=12, color=MUTED, weight=600)


def _agent_loop(c: Svg) -> None:
    """Centre panel: the decision loop itself, drawn as one closed cycle."""
    c.rect(520, 272, 880, 470, fill=NAVY, stroke=NAVY, radius=6)
    c.text(548, 302, "AGENT RUNTIME", size=12, color="#9DB2CF", weight=700)
    c.text(548, 336, "决策循环", size=24, color=WHITE, weight=700)
    c.text(
        548,
        366,
        "每一轮只决定一件事：下一个动作是什么",
        size=13,
        color="#B8C7DB",
    )

    cx, cy, rx, ry = 960, 545, 285, 142
    nodes = [
        (90, "观察状态", "读取工作空间里的事实", "#DCE7F5"),
        (20, "制定计划", "缺什么产物，就补什么", "#DCE7F5"),
        (-45, "规则决策", "查 rules，得到动作名", ORANGE),
        (-135, "调用工具", "只调探活成功的能力", "#DCE7F5"),
        (-205, "验证结果", "过 gates，才算完成", "#72B69E"),
    ]

    def point(angle: float) -> tuple[float, float]:
        rad = math.radians(angle)
        return cx + rx * math.cos(rad), cy - ry * math.sin(rad)

    boxes = []
    for angle, _, _, _ in nodes:
        px, py = point(angle)
        boxes.append((px - 108, py - 48, px + 108, py + 48))

    def blocked(p: tuple[float, float]) -> bool:
        return any(
            x1 <= p[0] <= x2 and y1 <= p[1] <= y2 for x1, y1, x2, y2 in boxes
        )

    # Walk the ellipse clockwise and keep only the visible arcs between nodes,
    # so every arrowhead lands exactly on a node border.
    run: list[tuple[float, float]] = []
    for step in range(0, 181):
        p = point(90 - step * 2)
        if blocked(p):
            if len(run) > 3:
                c.path(run, color="#5C7EA8", sw=2, arrow=True)
            run = []
        else:
            run.append(p)
    if len(run) > 3:
        c.path(run, color="#5C7EA8", sw=2, arrow=True)

    for angle, title, body, accent in nodes:
        px, py = point(angle)
        c.rect(
            px - 100,
            py - 40,
            200,
            80,
            fill="#123A73",
            stroke="#2F5C97",
            radius=4,
            sw=1,
        )
        c.rect(px - 100, py - 40, 4, 80, fill=accent, stroke=accent, radius=0)
        c.text(px - 82, py - 12, title, size=18, color=WHITE, weight=700)
        c.text(px - 82, py + 17, body, size=12, color="#B8C7DB")

    c.text(cx, cy - 14, "目标驱动", size=25, color=WHITE, weight=700, anchor="middle")
    c.text(
        cx,
        cy + 20,
        "而不是步骤驱动",
        size=14,
        color="#B8C7DB",
        anchor="middle",
    )

    c.line(548, 700, 1372, 700, color="#2F5C97", sw=1)
    c.text(548, 722, "循环退出条件", size=12, color=ORANGE, weight=700)
    c.text(668, 722, "所有分镜都有产物，且全部门禁通过", size=13, color="#DCE7F5")


def _agent_workspace(c: Svg) -> None:
    """Left panel: the shared, on-disk workspace the Agent reads and writes."""
    c.rect(70, 272, 370, 470, fill=WHITE, stroke=LINE, radius=6, sw=1)
    c.text(96, 302, "01 / WORKSPACE", size=12, color=ORANGE, weight=700)
    c.text(96, 338, "任务工作空间", size=22, color=NAVY, weight=700)
    c.text(96, 366, "Agent 的记忆：全部产物落盘，可读可回看", size=13, color=MUTED)
    c.line(96, 388, 414, 388, color="#E3E9F1", sw=1)

    assets = [
        "商品信息卡",
        "爆款骨架 + 逐镜分镜表",
        "新商品剧本",
        "用户素材片段库",
        "逐镜匹配结果",
        "门禁与质检报告",
        "任务状态与决策历史",
    ]
    for index, name in enumerate(assets):
        y = 418 + index * 40
        c.rect(98, y - 4, 8, 8, fill=BLUE, stroke=BLUE, radius=0)
        c.text(120, y, name, size=17, color=INK)

    c.line(96, 706, 414, 706, color="#E3E9F1", sw=1)
    c.text(96, 726, "每一轮都从这里读事实，也把结果写回这里", size=12, color=MUTED)

    c.text(480, 434, "读取", size=12, color=MUTED, weight=600, anchor="middle")
    c.line(442, 452, 518, 452, color=BLUE, sw=2, arrow=True)
    c.text(480, 572, "写回", size=12, color=MUTED, weight=600, anchor="middle")
    c.line(518, 590, 442, 590, color=BLUE, sw=2, arrow=True)


def agent_core() -> None:
    c = Svg("ViralForge：面向爆款复刻的多模态 Agent")
    c.header(
        "ViralForge：面向爆款复刻的多模态 Agent",
        "围绕复刻目标持续感知、规划、行动、观察结果，并为每个镜头重新选择下一动作",
        tag="VIRALFORGE · MULTIMODAL AGENT",
    )

    # Left: the actual task state that the Agent observes and updates.
    c.rect(70, 190, 365, 675, fill=WHITE, stroke=LINE, radius=6, sw=1)
    c.text(98, 222, "任务状态 / 记忆", size=12, color=ORANGE, weight=700)
    c.text(98, 258, "复刻任务工作空间", size=24, color=NAVY, weight=700)
    c.text(98, 288, "每轮读状态，行动后写回产物", size=14, color=MUTED)
    c.line(98, 310, 407, 310, color=LINE, sw=1)
    workspace = [
        ("输入", "商品信息卡"),
        ("参考", "爆款骨架 / 逐镜分镜表"),
        ("创作", "新商品剧本"),
        ("素材", "用户素材片段库"),
        ("进度", "逐镜匹配结果 / 任务状态"),
        ("反馈", "生成片段 / 门禁报告 / 历史"),
    ]
    for idx, (label, value) in enumerate(workspace):
        y = 350 + idx * 62
        c.text(98, y, label, size=12, color=BLUE, weight=700)
        c.text(158, y, value, size=16, color=INK, weight=600)
        if idx < len(workspace) - 1:
            c.line(98, y + 25, 407, y + 25, color="#E8EDF3", sw=1)
    c.text(98, 810, "当前观察示例", size=12, color=ORANGE_DARK, weight=700)
    c.lines(98, 836, ["第 4 镜：画面匹配 0.86", "口播不匹配 · BGM 存在 · TTS 可用"], size=13, color=MUTED, gap=22)

    # Centre: one real observe-plan-act-verify loop, with ViralForge actions.
    c.rect(475, 190, 970, 675, fill=NAVY, stroke=NAVY, radius=6)
    c.text(510, 222, "VIRALFORGE AGENT RUNTIME", size=12, color="#9DB2CF", weight=700)
    c.text(510, 258, "当前镜头的下一动作选择", size=27, color=WHITE, weight=700)
    c.text(510, 288, "不是按步骤号推进，而是依据当前状态持续再决策", size=14, color="#B8C7DB")
    cx, cy = 960, 535
    positions = [
        (960, 360, "观察状态", "当前镜头 / 已有产物", "#DCE7F5"),
        (1195, 455, "规划缺口", "口播缺失，画面可复用", "#DCE7F5"),
        (1110, 665, "策略决策", "rules 选择 keep + TTS", ORANGE),
        (810, 665, "行动执行", "静音、配音、铺 BGM", "#DCE7F5"),
        (725, 455, "结果观察", "gates 检查真实产物", "#72B69E"),
    ]
    for a, b in zip(positions, positions[1:] + positions[:1]):
        c.line(a[0], a[1], b[0], b[1], color="#5C7EA8", sw=2, arrow=True)
    for x, y, title, body, accent in positions:
        c.rect(x - 112, y - 43, 224, 86, fill="#123A73", stroke="#3B659B", radius=4, sw=1)
        c.rect(x - 112, y - 43, 5, 86, fill=accent, stroke=accent, radius=0)
        c.text(x - 88, y - 13, title, size=18, color=WHITE, weight=700)
        c.text(x - 88, y + 17, body, size=12, color="#B8C7DB")
    c.circle(cx, cy, 87, fill="#0B2A5B", stroke="#3B659B", sw=1)
    c.text(cx, cy - 14, "目标", size=18, color=ORANGE, weight=700, anchor="middle")
    c.text(cx, cy + 17, "复刻爆款结构", size=15, color=WHITE, weight=700, anchor="middle")
    c.text(510, 798, "未通过：更新工作空间 → 重新规划 → 重试 / 降级", size=14, color="#F7C28B", weight=700)

    # Right: rules and gates are the policy layer around actions, not a step.
    c.rect(1485, 190, 365, 675, fill=WHITE, stroke=LINE, radius=6, sw=1)
    c.text(1513, 222, "策略与护栏", size=12, color=ORANGE, weight=700)
    c.text(1513, 258, "Agent Policy Layer", size=23, color=NAVY, weight=700)
    c.text(1513, 288, "决定做什么、先做什么、失败怎么退", size=13, color=MUTED)
    c.line(1513, 310, 1822, 310, color=LINE, sw=1)
    policy = [
        ("rules", ORANGE, ["画面 ≥ 0.80：直接复用", "0.55–0.80：编辑后使用", "更低：进入补片 / 降级"]),
        ("tools", BLUE, ["动作语义稳定", "按探活结果选择实现", "Gemini / Qwen / ffmpeg / TTS"]),
        ("gates", GREEN, ["实测音频、字幕、画面", "不通过：淘汰或回判", "通过后才进入合成"]),
        ("安全约束", NAVY, ["真人风控 → 线稿替身", "显式失败，不静默丢内容"]),
    ]
    for idx, (title, accent, lines) in enumerate(policy):
        y = 350 + idx * 116
        c.rect(1513, y, 42, 3, fill=accent, stroke=accent, radius=0)
        c.text(1513, y + 27, title, size=19, color=NAVY, weight=700)
        c.lines(1513, y + 56, lines, size=13, color=MUTED, gap=21)

    c.rect(70, 895, 1780, 42, fill=SOFT, stroke=LINE, radius=4, sw=1)
    c.text(98, 922, "能力池", size=14, color=NAVY, weight=700)
    c.text(190, 922, "理解 Gemini / Qwen   ·   生成 Seedream / Seedance   ·   媒体 ffmpeg / demucs / librosa   ·   外挂 TTS / 字幕克隆 / ASR", size=13, color=INK)
    c.footer("ViralForge 的 Agent 以逐镜任务为单位：观察状态，选择动作，验证结果，直到整条成片可交付。")
    c.render("vf00_agent_core")


def overview() -> None:
    c = Svg("一个爆款视频，如何被拆解并复刻成你的商品视频")
    c.header(
        "一个爆款视频，如何被拆解并复刻成你的商品视频",
        "受控 Agent 读取事实、执行规则、调度能力，将爆款结构稳定迁移到不同商品",
        tag="CONTROLLED AGENT · USER MATERIAL FIRST",
    )

    # The Agent is the only heavy visual block: a control tower, not a sidebar menu.
    c.rect(70, 185, 320, 735, fill=NAVY, stroke=NAVY, radius=6)
    c.text(105, 220, "CONTROL PLANE", size=12, color="#9DB2CF", weight=700)
    c.text(105, 265, "Agent 中枢", size=31, color=WHITE, weight=700)
    c.lines(
        105,
        305,
        ["读取任务状态与结构化事实", "在安全边界内决定下一动作"],
        size=14,
        color="#C7D5E7",
        gap=24,
    )

    agent_steps = [
        (370, "读取状态", "task / capability status", "#DCE7F5"),
        (455, "结构化事实", "reference / product / material", "#DCE7F5"),
        (540, "rules 决策", "动作名 + 命中痕迹", ORANGE),
        (625, "tools 调度", "调用已探活能力", "#DCE7F5"),
        (710, "gates 验收", "真实产物必须通过", "#72B69E"),
    ]
    c.line(118, 370, 118, 710, color="#49698F", sw=2)
    for y, title, body, color in agent_steps:
        c.circle(118, y, 7, fill=color, stroke=color)
        c.text(145, y - 8, title, size=18, color=WHITE, weight=700)
        c.text(145, y + 20, body, size=12, color="#9DB2CF", weight=400)

    c.line(105, 770, 355, 770, color="#49698F", sw=1)
    c.text(105, 804, "未通过", size=13, color=ORANGE, weight=700)
    c.text(178, 804, "修正事实  >  重判  >  重试 / 降级", size=13, color="#FFD5AA")
    c.text(105, 862, "受控，而非自由自治", size=16, color=WHITE, weight=700)
    c.text(105, 890, "每次决策、调用与降级都可追溯", size=12, color="#9DB2CF")

    stage_xs = [445, 920, 1395]
    stage_w = 410
    stage_centers = [x + stage_w / 2 for x in stage_xs]

    # One restrained control rail links the Agent to all three value-chain stages.
    c.line(390, 204, 1805, 204, color=ORANGE, sw=1.5)
    for center in stage_centers:
        c.circle(center, 204, 4, fill=ORANGE, stroke=ORANGE)
        c.line(center, 208, center, 230, color=ORANGE, sw=1.5)

    c.line(892, 235, 892, 910, color="#E1E6ED", sw=1)
    c.line(1367, 235, 1367, 910, color="#E1E6ED", sw=1)
    for center in (892, 1367):
        c.line(center - 14, 510, center, 520, color="#A8B4C3", sw=1.5)
        c.line(center, 520, center - 14, 530, color="#A8B4C3", sw=1.5)

    stages = [
        {
            "number": "01",
            "title": "理解爆款",
            "headline": "把“为什么火”变成可执行结构",
            "items": [
                ("爆款参考片", "读取真实画面、声音与时长"),
                ("逐镜拆解", "镜头语言 / 台词 / 情绪 / 叙事"),
                ("可复用骨架", "钩子 / 节奏 / 反转 / 植入位置"),
            ],
            "output": "传播骨架",
            "note": "按镜拆，让匹配与验收拥有统一粒度",
        },
        {
            "number": "02",
            "title": "复刻内容",
            "headline": "保留传播机制，替换商品表达",
            "items": [
                ("商品事实", "用户输入优先，模型推断单列"),
                ("剧本映射", "保留结构，替换人物、场景与商品"),
                ("素材匹配", "画面与口播分开评分"),
            ],
            "output": "可执行分镜",
            "note": "rules 决定直接裁剪 / 素材编辑 / AI 补片",
        },
        {
            "number": "03",
            "title": "组装成片",
            "headline": "混合真实素材与补片，统一交付",
            "items": [
                ("分段出片", "真实素材优先，AI 只补缺口"),
                ("音轨与字幕", "配音、BGM、字幕统一时间轴"),
                ("质量验收", "利用率 / 响度 / 重复 / 门禁对账"),
            ],
            "output": "可交付成片",
            "note": "gates 对真实产物验收，不接受“AI 说可用”",
        },
    ]

    for x, stage in zip(stage_xs, stages):
        c.text(x, 252, stage["number"], size=14, color=ORANGE, weight=700)
        c.text(x, 292, stage["title"], size=27, color=NAVY, weight=700)
        c.text(x, 330, stage["headline"], size=14, color=MUTED)
        c.line(x, 356, x + stage_w, 356, color="#DDE3EB", sw=1)

        item_ys = [410, 525, 640]
        c.line(x + 8, item_ys[0], x + 8, item_ys[-1], color="#CFD7E2", sw=1.5)
        for y, (title, body) in zip(item_ys, stage["items"]):
            c.circle(x + 8, y, 5, fill=NAVY, stroke=NAVY)
            c.text(x + 32, y - 8, title, size=19, color=INK, weight=700)
            c.text(x + 32, y + 22, body, size=13, color=MUTED)

        c.rect(
            x,
            745,
            stage_w,
            86,
            fill="#F3F6FA",
            stroke="#F3F6FA",
            radius=3,
        )
        c.text(x + 22, 771, "阶段产出", size=12, color=MUTED, weight=700)
        c.text(x + 22, 805, stage["output"], size=21, color=NAVY, weight=700)
        c.line(x, 872, x + 30, 872, color=ORANGE, sw=2)
        c.text(x + 42, 872, stage["note"], size=12, color=ORANGE_DARK, weight=600)

    c.footer("保留爆款的传播结构，替换商品内容；rules 决策，tools 执行，gates 验收。")
    c.render("vf01_overview")


def decomposition_mapping() -> None:
    c = Svg("爆款如何拆解，并映射成新商品剧本")
    c.header(
        "爆款如何拆解，并映射成新商品剧本",
        "复刻的对象不是原片画面，而是每一镜背后的时长、镜头语言和叙事作用",
        tag="按镜拆，才能逐镜匹配与验收",
    )

    bands = [
        (185, 220, "A", "参考片", "把原片切成\n连续镜头"),
        (430, 225, "B", "结构映射", "明确哪些保留\n哪些替换"),
        (680, 245, "C", "新剧本", "对应参考镜\n逐镜写新内容"),
    ]
    for y, h, number, title, subtitle in bands:
        c.rect(245, y, 1605, h, fill=SOFT, stroke="#E6ECF3", radius=14, sw=1)
        c.section_label(y, h, number, title, subtitle)

    shot_xs = [270, 583, 896, 1209, 1522]
    shot_w = 280
    reference = [
        ("参考镜 01", "开场钩子", "1.2s · 商品特写"),
        ("参考镜 02", "痛点铺垫", "2.4s · 中景跟拍"),
        ("参考镜 03", "功能演示", "3.0s · 近景推镜"),
        ("参考镜 04", "效果证明", "2.2s · 对比切换"),
        ("参考镜 05", "行动引导", "1.6s · 正面定镜"),
    ]
    rewritten = [
        ("新镜头 01", "新商品钩子", "仍为 1.2s · 特写"),
        ("新镜头 02", "目标用户痛点", "仍为 2.4s · 跟拍"),
        ("新镜头 03", "新功能演示", "仍为 3.0s · 推镜"),
        ("新镜头 04", "新效果证明", "仍为 2.2s · 对比"),
        ("新镜头 05", "新转化文案", "仍为 1.6s · 定镜"),
    ]

    for index, (x, values) in enumerate(zip(shot_xs, reference), start=1):
        title, function, detail = values
        c.card(
            x,
            220,
            shot_w,
            150,
            title,
            [function, detail],
            number=str(index),
            accent=ORANGE if index == 1 else BLUE,
            shadow=False,
            body_size=15,
            body_gap=24,
        )
        if index < len(shot_xs):
            c.line(
                x + shot_w + 8,
                295,
                shot_xs[index] - 10,
                295,
                color=LINE,
                sw=2,
                arrow=True,
            )

    c.card(
        270,
        470,
        735,
        145,
        "保留：爆款的传播骨架",
        ["分镜数量与时长 · 景别与运镜", "转场与节奏 · 钩子/铺垫/反转/转化"],
        accent=BLUE,
        fill=PALE_BLUE,
        shadow=False,
        body_size=17,
        body_gap=28,
    )
    c.card(
        1085,
        470,
        715,
        145,
        "替换：你的商品内容",
        ["人物与场景 · 商品与卖点", "商品相关台词 · 需要表达的具体动作"],
        accent=ORANGE,
        fill=PALE_ORANGE,
        shadow=False,
        body_size=17,
        body_gap=28,
    )
    c.line(1008, 542, 1077, 542, color=NAVY, sw=1.5)
    c.text(1043, 524, "逐镜对应", size=14, color=NAVY, weight=700, anchor="middle")

    for index, (x, values) in enumerate(zip(shot_xs, rewritten), start=1):
        title, function, detail = values
        c.card(
            x,
            720,
            shot_w,
            165,
            title,
            [f"对应参考镜 {index:02d}", function, detail],
            number=str(index),
            accent=ORANGE if index == 1 else BLUE_2,
            fill=PALE_BLUE_2,
            shadow=False,
            body_size=14,
            body_gap=23,
        )
        c.line(
            x + shot_w / 2,
            390,
            x + shot_w / 2,
            430,
            color="#AFC2DA",
            sw=2,
            dashed=True,
        )
        c.line(
            x + shot_w / 2,
            655,
            x + shot_w / 2,
            704,
            color=BLUE_2,
            sw=2,
            dashed=True,
            arrow=True,
        )

    c.footer("严格复刻保留镜头骨架并替换内容；创意仿写保留传播机制，允许剧情变化。")
    c.render("vf02_decompose_mapping")


def material_reuse() -> None:
    c = Svg("用户素材如何被匹配、复用和补齐")
    c.header(
        "用户素材如何被匹配、复用和补齐",
        "先把长视频切成可检索片段，再逐镜判断：直接用、参考编辑，还是由 AI 补片",
        tag="素材利用率是主指标",
    )

    rows = [
        (180, 205, "01", "素材建库", "按内容完整性切片\n建立语义索引"),
        (410, 275, "02", "逐镜决策", "画面和口播\n分开判断"),
        (710, 220, "03", "窗口复用", "同片段多次使用\n但不重复画面"),
    ]
    for y, h, number, title, subtitle in rows:
        c.rect(245, y, 1605, h, fill=SOFT, stroke="#E6ECF3", radius=14, sw=1)
        c.section_label(y, h, number, title, subtitle)

    c.card(
        270,
        215,
        300,
        135,
        "用户原始视频",
        ["一整卷连续画面", "可能包含多个动作与场景"],
        number="1",
        shadow=False,
    )
    c.card(
        660,
        215,
        300,
        135,
        "按语义切片",
        ["一个片段表达一个动作", "时间轴首尾连续覆盖"],
        number="2",
        shadow=False,
    )
    c.rect(1050, 205, 750, 155, fill=WHITE, stroke=LINE, radius=6, shadow=False)
    c.text(1080, 236, "可检索素材库", size=21, color=NAVY, weight=700)
    c.lines(
        1080,
        268,
        ["每段都有时间码、画面标签、人物/声音事实和召回文本"],
        size=15,
        color=MUTED,
        gap=22,
    )
    timeline_x = 1080
    timeline_y = 307
    segment_widths = [135, 170, 125, 180]
    segment_labels = ["M01 商品特写", "M02 真人使用", "M03 环境空镜", "M04 口播演示"]
    fills = [PALE_BLUE, "#DCE9F8", PALE_ORANGE, "#E3EDF9"]
    cursor = timeline_x
    for width, label, fill in zip(segment_widths, segment_labels, fills):
        c.rect(cursor, timeline_y, width, 34, fill=fill, stroke=WHITE, sw=2, radius=6)
        c.text(
            cursor + width / 2,
            timeline_y + 18,
            label,
            size=12,
            color=NAVY if fill != PALE_ORANGE else ORANGE_DARK,
            weight=600,
            anchor="middle",
        )
        cursor += width
    c.line(585, 282, 645, 282, color=BLUE, sw=2.5, arrow=True)
    c.line(975, 282, 1035, 282, color=BLUE, sw=2.5, arrow=True)

    c.card(
        270,
        462,
        315,
        175,
        "待填的新分镜",
        ["画面匹配：决定是否采用", "口播匹配：只决定音轨", "时长、主体、动作一起评分"],
        number="3",
        shadow=False,
        body_size=15,
        body_gap=25,
    )
    c.diamond(700, 548, 150, 150, fill=PALE_ORANGE, stroke=ORANGE, shadow=False)
    c.text(700, 527, "匹配度", size=21, color=ORANGE_DARK, weight=700, anchor="middle")
    c.text(700, 558, "0 ～ 1", size=18, color=ORANGE_DARK, weight=600, anchor="middle")
    c.text(700, 584, "规则表决策", size=14, color=MUTED, weight=400, anchor="middle")
    c.line(600, 548, 618, 548, color=ORANGE, sw=2.5, arrow=True, arrow_color="orange")

    c.rect(820, 438, 980, 215, fill=WHITE, stroke=LINE, radius=6, shadow=False)
    c.text(850, 466, "三种执行路线", size=21, color=NAVY, weight=700)
    route_xs = [850, 1165, 1480]
    route_data = [
        (
            "≥ 0.80",
            "直接裁剪",
            ["真素材直接出镜", "按分镜时长裁取"],
            BLUE,
            PALE_BLUE,
        ),
        (
            "0.55 ≤ 分数 < 0.80",
            "素材编辑",
            ["画面可用但需改造", "作为参考视频生成"],
            ORANGE,
            PALE_ORANGE,
        ),
        (
            "< 0.55",
            "AI 重新生成",
            ["素材库无法表达此镜", "用设定图与商品图补齐"],
            MUTED,
            "#F1F4F8",
        ),
    ]
    for x, (score, title, body, accent, fill) in zip(route_xs, route_data):
        c.rect(x, 492, 280, 135, fill=fill, stroke=accent, sw=1.2, radius=6)
        c.rect(x + 18, 507, 42, 3, fill=accent, stroke=accent, radius=0)
        c.text(x + 18, 528, score, size=13, color=accent, weight=700)
        c.text(x + 18, 561, title, size=20, color=NAVY, weight=700)
        c.lines(x + 18, 588, body, size=13, color=MUTED, gap=21)
    c.line(777, 548, 805, 548, color=ORANGE, sw=2.5, arrow=True, arrow_color="orange")

    c.rect(270, 750, 1120, 145, fill=WHITE, stroke=LINE, radius=6, shadow=False)
    c.text(300, 780, "同一素材片段的取用窗口", size=21, color=NAVY, weight=700)
    bar_x = 300
    bar_y = 824
    total_w = 1055
    windows = [
        (250, "镜头 1 · 第一刀", BLUE),
        (300, "镜头 3 · 第二刀", BLUE_2),
        (205, "镜头 6 · 第三刀", ORANGE),
        (300, "尚未使用", "#DCE3EB"),
    ]
    cursor = bar_x
    for width, label, fill in windows:
        c.rect(cursor, bar_y, width, 46, fill=fill, stroke=WHITE, sw=2, radius=7)
        c.text(
            cursor + width / 2,
            bar_y + 24,
            label,
            size=14,
            color=WHITE if fill != "#DCE3EB" else MUTED,
            weight=600,
            anchor="middle",
        )
        cursor += width
    c.line(bar_x, 881, bar_x + total_w, 881, color=LINE, sw=2)
    c.text(bar_x, 901, "0s", size=12, color=MUTED)
    c.text(bar_x + total_w, 901, "片段末尾", size=12, color=MUTED, anchor="end")

    c.card(
        1450,
        750,
        350,
        145,
        "窗口规则",
        ["从上次结束处继续取", "优先未使用的新鲜画面", "不足才回卷，并显式留痕"],
        accent=ORANGE,
        fill=PALE_ORANGE,
        shadow=False,
        body_size=15,
        body_gap=23,
    )

    c.footer("画面过关就用画面，声音问题交给声音解决；素材确实覆盖不了时才由 AI 补片。")
    c.render("vf03_material_reuse")


def architecture_quality() -> None:
    c = Svg("通用爆款复刻 Agent：决策与能力调度")
    c.header(
        "通用爆款复刻 Agent：决策与能力调度",
        "一个稳定控制面，向上接收事实、向右调用能力、向下沉淀可复用的设计基线",
        tag="POLICY-DRIVEN · TOOL-AUGMENTED · VERIFIABLE",
    )

    # Fact layer: four inputs share one protocol and one visual band.
    c.text(70, 192, "01 / FACT LAYER", size=12, color=ORANGE, weight=700)
    c.text(70, 226, "统一事实输入", size=24, color=NAVY, weight=700)
    c.text(70, 258, "变化的任务，被归一为稳定字段", size=13, color=MUTED)
    c.rect(300, 180, 1550, 120, fill="#F7F9FC", stroke="#F7F9FC", radius=4)

    fact_xs = [330, 710, 1090, 1470]
    facts = [
        ("参考片事实", "分镜 / 节奏 / 钩子 / 音轨"),
        ("商品事实", "商品图 / 卖点 / 限制 / 推断"),
        ("素材事实", "人物 / 口播 / BGM / 匹配度"),
        ("运行事实", "步骤状态 / 工具探活 / 失败记录"),
    ]
    for index, (x, (title, body)) in enumerate(zip(fact_xs, facts)):
        if index:
            c.line(x - 20, 202, x - 20, 278, color="#DDE3EB", sw=1)
        c.text(x, 218, title, size=18, color=NAVY, weight=700)
        c.text(x, 258, body, size=12, color=MUTED)
        c.line(x + 145, 300, x + 145, 334, color="#CFD7E2", sw=1.5)

    # The control plane is the sole dominant object.
    c.rect(220, 350, 1120, 380, fill=NAVY, stroke=NAVY, radius=6)
    c.text(255, 385, "02 / CONTROL PLANE", size=12, color="#9DB2CF", weight=700)
    c.text(255, 428, "通用爆款复刻 Agent", size=30, color=WHITE, weight=700)
    c.text(
        255,
        466,
        "固定九步主链承载状态；模型提供事实，规则决定动作，工具负责执行",
        size=14,
        color="#C7D5E7",
    )

    control_xs = [320, 540, 760, 980, 1200]
    control_nodes = [
        ("感知事实", "LLM / VLM", "#DCE7F5"),
        ("rules 决策", "动作名 + 痕迹", ORANGE),
        ("状态计划", "pipeline / task", "#DCE7F5"),
        ("tools 调度", "action / registry", "#DCE7F5"),
        ("gates 验收", "media metrics", "#72B69E"),
    ]
    c.line(control_xs[0], 555, control_xs[-1], 555, color="#577599", sw=2)
    for x, (title, note, color) in zip(control_xs, control_nodes):
        c.circle(x, 555, 9, fill=color, stroke=color)
        c.text(x, 520, title, size=17, color=WHITE, weight=700, anchor="middle")
        c.text(x, 590, note, size=12, color="#9DB2CF", anchor="middle")

    c.path(
        [(1200, 616), (1200, 650), (540, 650), (540, 616)],
        color=ORANGE,
        sw=1.6,
        dashed=True,
        arrow=True,
        arrow_color="orange",
    )
    c.text(
        870,
        674,
        "未通过：修正事实  >  重新决策  >  重试或降级",
        size=13,
        color="#FFD5AA",
        weight=600,
        anchor="middle",
    )
    c.text(
        255,
        706,
        "受控边界：模型不自由尝试任意工具；所有调用、命中规则和降级路径均留痕。",
        size=12,
        color="#B8C7DB",
    )

    # Capability rack: one rack, four replaceable implementations, no mini-dashboard cards.
    c.rect(1415, 350, 435, 380, fill="#F7F9FC", stroke="#F7F9FC", radius=4)
    c.text(1450, 385, "03 / CAPABILITY RACK", size=12, color=ORANGE, weight=700)
    c.text(1450, 428, "可插拔 Tools", size=27, color=NAVY, weight=700)
    c.text(1450, 462, "动作名稳定，底层实现可替换", size=13, color=MUTED)

    tool_rows = [
        (510, "理解模型", "Gemini / Qwen"),
        (565, "生成模型", "Seedream / Seedance"),
        (620, "媒体工具", "ffmpeg / demucs"),
        (675, "外挂能力", "TTS / 字幕 / ASR"),
    ]
    for index, (y, title, implementation) in enumerate(tool_rows):
        if index:
            c.line(1450, y - 28, 1815, y - 28, color="#E1E6ED", sw=1)
        c.text(1450, y, title, size=17, color=INK, weight=700)
        c.text(1815, y, implementation, size=12, color=MUTED, anchor="end")

    c.text(1378, 462, "调用", size=11, color=ORANGE_DARK, weight=700, anchor="middle")
    c.line(1340, 482, 1402, 482, color=ORANGE, sw=1.8, arrow=True, arrow_color="orange")
    c.text(1378, 620, "结果", size=11, color=BLUE, weight=700, anchor="middle")
    c.line(1402, 640, 1340, 640, color=BLUE, sw=1.8, arrow=True)

    # Design baseline: strategic conclusions, not another feature-card row.
    c.line(70, 780, 1850, 780, color="#DDE3EB", sw=1)
    c.text(70, 812, "04 / DESIGN BASELINE", size=12, color=ORANGE, weight=700)
    c.text(70, 850, "通用性的来源", size=24, color=NAVY, weight=700)

    baseline_xs = [360, 730, 1100, 1470]
    baselines = [
        ("统一事实协议", "不同商品与素材，归一成相同字段"),
        ("规则可配置", "阈值、优先级和兜底独立演进"),
        ("工具可插拔", "能力实现替换，不改上层动作语义"),
        ("任务可追溯", "状态、决策、调用、门禁完整留痕"),
    ]
    for index, (x, (title, body)) in enumerate(zip(baseline_xs, baselines)):
        if index:
            c.line(x - 25, 812, x - 25, 904, color="#DDE3EB", sw=1)
        c.text(x, 836, title, size=19, color=NAVY, weight=700)
        c.text(x, 876, body, size=12, color=MUTED)

    c.footer("通用性不来自“万能模型”，而来自稳定流程、可配置 rules、可插拔 tools 和可验证闭环。")
    c.render("vf04_architecture_quality")


def _style(stroke: str, background: str, *, font_size: int = 16, text: str = INK) -> dict:
    return {
        "stroke_color": stroke,
        "background_color": background,
        "fill_style": "solid",
        "stroke_width": 2,
        "font_size": font_size,
        "text_color": text,
    }


def _node(
    node_id: str,
    label: str,
    *,
    shape: str = "rounded_rect",
    kind: str = "blue",
) -> dict:
    styles = {
        "blue": _style(BLUE, PALE_BLUE),
        "navy": _style(NAVY, "#DCE8F7"),
        "orange": _style(ORANGE_DARK, PALE_ORANGE),
        "green": _style(GREEN, PALE_GREEN),
        "gray": _style(MUTED, "#F1F4F8"),
    }
    return {
        "id": node_id,
        "label": label,
        "shape": shape,
        "style": styles[kind],
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    label: str = "",
    *,
    color: str = BLUE,
    dashed: bool = False,
) -> dict:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "label": label,
        "style": {
            "stroke_color": color,
            "stroke_width": 2,
            "stroke_style": "dashed" if dashed else "solid",
            "end_arrowhead": "arrow",
        },
    }


def _graph_specs() -> dict[str, dict]:
    return {
        "vf00_agent_core": {
            "metadata": {
                "title": "爆款复刻 Agent 的业务决策链",
                "diagram_type": "flowchart",
                "direction": "TB",
                "ranksep": 0.62,
                "nodesep": 0.45,
                "background_color": WHITE,
                "output_dir": str(SOURCE_DIR),
            },
            "nodes": [
                _node("product", "商品信息卡", shape="cylinder", kind="blue"),
                _node("reference", "爆款参考片", shape="ellipse", kind="navy"),
                _node("materials", "用户素材片段库", shape="cylinder", kind="blue"),
                _node("facts", "多模态事实层", shape="cylinder", kind="blue"),
                _node("skeleton", "爆款骨架与逐镜分镜表"),
                _node("script", "新商品剧本", kind="green"),
                _node("agent", "Agent：逐镜复刻决策", shape="ellipse", kind="navy"),
                _node("rules", "rules：动作选择", shape="diamond", kind="orange"),
                _node("tools", "工具调用：复用 / 编辑 / 补片", shape="cylinder", kind="orange"),
                _node("gates", "gates：真实产物验收", shape="diamond", kind="green"),
                _node("compose", "组装成片", kind="navy"),
                _node("deliver", "可追溯复刻成片", shape="ellipse", kind="green"),
            ],
            "edges": [
                _edge("e1", "product", "facts", "商品事实"),
                _edge("e2", "reference", "facts", "参考片事实"),
                _edge("e3", "materials", "facts", "素材事实"),
                _edge("e4", "facts", "skeleton", "拆解"),
                _edge("e5", "skeleton", "script", "结合商品改写", color=ORANGE),
                _edge("e6", "script", "agent", "逐镜目标", color=NAVY),
                _edge("e7", "facts", "agent", "当前镜头状态"),
                _edge("e8", "agent", "rules", "查规则", color=ORANGE),
                _edge("e9", "rules", "tools", "keep / edit / generate", color=ORANGE),
                _edge("e10", "tools", "gates", "镜头产物"),
                _edge("e11", "gates", "compose", "通过", color=GREEN),
                _edge("e12", "gates", "agent", "失败：回判", color=ORANGE, dashed=True),
                _edge("e13", "compose", "deliver", "全片组装", color=GREEN),
            ],
        },
        "vf01_overview": {
            "metadata": {
                "title": "爆款复刻总览",
                "diagram_type": "flowchart",
                "direction": "TB",
                "ranksep": 0.72,
                "nodesep": 0.48,
                "background_color": WHITE,
                "output_dir": str(SOURCE_DIR),
            },
            "nodes": [
                _node("reference", "爆款参考片", shape="ellipse", kind="navy"),
                _node("decompose", "逐镜拆解"),
                _node("skeleton", "传播骨架"),
                _node("rewrite", "新剧本映射"),
                _node("materials", "用户素材库", shape="cylinder"),
                _node("match", "逐镜匹配"),
                _node("state", "任务状态", shape="cylinder", kind="blue"),
                _node("agent", "Agent中枢", shape="ellipse", kind="navy"),
                _node("decision", "rules决策", shape="diamond", kind="orange"),
                _node("crop", "直接裁剪", kind="blue"),
                _node("edit", "素材编辑", kind="orange"),
                _node("generate", "AI补片", kind="gray"),
                _node("compose", "组装成片", kind="navy"),
                _node("review", "质检交付", shape="ellipse", kind="green"),
            ],
            "edges": [
                _edge("e1", "reference", "decompose"),
                _edge("e2", "decompose", "skeleton"),
                _edge("e3", "skeleton", "rewrite"),
                _edge("e4", "rewrite", "match"),
                _edge("e5", "materials", "match", "优先使用"),
                _edge("e6", "match", "agent", "事实"),
                _edge("e6a", "state", "agent", "当前步骤"),
                _edge("e6b", "agent", "decision", "查规则", color=ORANGE),
                _edge("e7", "decision", "crop", "≥0.80"),
                _edge("e8", "decision", "edit", "0.55≤s<0.80", color=ORANGE),
                _edge("e9", "decision", "generate", "<0.55", color=MUTED),
                _edge("e10", "crop", "compose"),
                _edge("e11", "edit", "compose"),
                _edge("e12", "generate", "compose"),
                _edge("e13", "compose", "review"),
            ],
        },
        "vf02_decompose_mapping": {
            "metadata": {
                "title": "爆款拆解与映射",
                "diagram_type": "flowchart",
                "direction": "LR",
                "ranksep": 0.82,
                "nodesep": 0.50,
                "background_color": WHITE,
                "output_dir": str(SOURCE_DIR),
            },
            "nodes": [
                _node("reference", "爆款原片", shape="ellipse", kind="navy"),
                _node("shots", "逐镜切分"),
                _node("camera", "镜头语言"),
                _node("narrative", "叙事结构"),
                _node("rhythm", "节奏情绪"),
                _node("audio", "声音风格"),
                _node("mapping", "结构映射", shape="diamond", kind="orange"),
                _node("product", "商品事实卡", shape="cylinder"),
                _node("strict", "严格复刻"),
                _node("creative", "创意仿写", kind="orange"),
                _node("script", "新商品剧本", shape="ellipse", kind="green"),
            ],
            "edges": [
                _edge("e1", "reference", "shots"),
                _edge("e2", "shots", "camera"),
                _edge("e3", "shots", "narrative"),
                _edge("e4", "shots", "rhythm"),
                _edge("e5", "shots", "audio"),
                _edge("e6", "camera", "mapping"),
                _edge("e7", "narrative", "mapping"),
                _edge("e8", "rhythm", "mapping"),
                _edge("e9", "audio", "mapping"),
                _edge("e10", "product", "mapping", "替换内容", color=ORANGE),
                _edge("e11", "mapping", "strict", "一镜对应"),
                _edge("e12", "mapping", "creative", "机制对应", color=ORANGE),
                _edge("e13", "strict", "script"),
                _edge("e14", "creative", "script"),
            ],
        },
        "vf03_material_reuse": {
            "metadata": {
                "title": "素材匹配与复用",
                "diagram_type": "flowchart",
                "direction": "TB",
                "ranksep": 0.72,
                "nodesep": 0.48,
                "background_color": WHITE,
                "output_dir": str(SOURCE_DIR),
            },
            "nodes": [
                _node("raw", "用户原视频", shape="ellipse", kind="navy"),
                _node("cut", "语义切片"),
                _node("library", "素材片段库", shape="cylinder"),
                _node("shot", "待填分镜"),
                _node("score", "匹配度", shape="diamond", kind="orange"),
                _node("crop", "直接裁剪"),
                _node("edit", "素材编辑", kind="orange"),
                _node("generate", "AI生成", kind="gray"),
                _node("window", "取用窗口"),
                _node("voice", "声音另处理", kind="orange"),
                _node("segment", "混合出片", shape="ellipse", kind="green"),
            ],
            "edges": [
                _edge("e1", "raw", "cut"),
                _edge("e2", "cut", "library"),
                _edge("e3", "library", "score"),
                _edge("e4", "shot", "score"),
                _edge("e5", "score", "crop", "≥0.80"),
                _edge("e6", "score", "edit", "0.55≤s<0.80", color=ORANGE),
                _edge("e7", "score", "generate", "<0.55", color=MUTED),
                _edge("e8", "crop", "window"),
                _edge("e9", "window", "voice", "口播不符", color=ORANGE),
                _edge("e10", "window", "segment"),
                _edge("e11", "voice", "segment"),
                _edge("e12", "edit", "segment"),
                _edge("e13", "generate", "segment"),
            ],
        },
        "vf04_architecture_quality": {
            "metadata": {
                "title": "通用Agent决策调度",
                "diagram_type": "flowchart",
                "direction": "TB",
                "ranksep": 0.72,
                "nodesep": 0.45,
                "background_color": WHITE,
                "output_dir": str(SOURCE_DIR),
            },
            "nodes": [
                _node("inputs", "任务输入", shape="ellipse", kind="navy"),
                _node("observe", "提取事实"),
                _node("rules", "规则决策", shape="diamond", kind="orange"),
                _node("state", "任务状态", shape="cylinder", kind="blue"),
                _node("agent", "Agent中枢", shape="ellipse", kind="navy"),
                _node("understand", "理解模型"),
                _node("generate", "生成模型", kind="orange"),
                _node("media", "媒体工具"),
                _node("plugins", "外挂能力", kind="orange"),
                _node("gates", "门禁验收", shape="diamond", kind="orange"),
                _node("output", "成片交付", shape="ellipse", kind="green"),
            ],
            "edges": [
                _edge("e1", "inputs", "observe"),
                _edge("e2", "observe", "rules", "结构化事实"),
                _edge("e3", "rules", "agent", "动作名", color=ORANGE),
                _edge("e4", "state", "agent", "当前步骤"),
                _edge("e5", "agent", "understand", "调度"),
                _edge("e6", "agent", "generate", "调度", color=ORANGE),
                _edge("e7", "agent", "media", "调度"),
                _edge("e8", "agent", "plugins", "调度", color=ORANGE),
                _edge("e9", "understand", "gates"),
                _edge("e10", "generate", "gates"),
                _edge("e11", "media", "gates"),
                _edge("e12", "plugins", "gates"),
                _edge("e13", "gates", "output", "通过", color=GREEN),
                _edge("e14", "gates", "rules", "失败重判", color=ORANGE, dashed=True),
            ],
        },
    }


def write_graph_specs() -> None:
    for stem, graph in _graph_specs().items():
        path = SOURCE_DIR / f"{stem}.graph.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(graph, file, ensure_ascii=False, indent=2)
        print(f"graph source: {path.name}")


def main() -> None:
    agent_core()
    overview()
    decomposition_mapping()
    material_reuse()
    architecture_quality()
    write_graph_specs()


if __name__ == "__main__":
    main()
