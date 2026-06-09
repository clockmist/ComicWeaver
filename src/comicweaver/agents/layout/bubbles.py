"""Dialogue-bubble placement engine.

Given panel bboxes, dialogue content, and bubble hints from the storyboard,
this module computes the final (x, y, w, h) for each speech/thought/narration
bubble in normalised page coordinates.

Design principle:
    BubbleHint from StoryboardAgent is treated as a **suggestion for the
    ImageAgent** (so it knows to leave space when generating the panel image).
    The LayoutAgent uses BubbleHint as a starting candidate but may adjust
    positions based on text length, panel boundaries, and multi-bubble
    stacking.  A future VLM path will additionally analyse the actual image
    to avoid faces and focal regions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from comicweaver.core.schema import BoundingBox, BubbleHint

if TYPE_CHECKING:
    from comicweaver.core.schema import Dialogue, PanelImage, PanelPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bubble placement result
# ---------------------------------------------------------------------------

@dataclass
class BubblePlacement:
    """Finalised bubble coordinates (normalised 0-1, relative to full page)."""
    panel_id: str
    dialogue_index: int
    bubble_type: str       # "speech" | "thought" | "shout" | "whisper" | "narration"
    speaker: str
    text: str
    x: float
    y: float
    w: float
    h: float
    occlusion_score: float = 0.0
    tail_direction: str = "auto"  # "up" | "down" | "left" | "right" | "up_left" | ...
    font_size_pt: int = 12
    style: dict = field(default_factory=dict)  # {fill, outline, border, radius, tail}


def _derive_tail_direction(
    bubble_center_x: float, bubble_center_y: float,
    panel_bbox: BoundingBox,
) -> str:
    """Determine which way the bubble's tail should point.

    The tail points from the bubble's EDGE toward the panel CENTER
    (where the speaker is most likely located).  So a bubble in the
    top-left corner gets a tail pointing "down_right".
    """
    pcx = panel_bbox.x + panel_bbox.width / 2
    pcy = panel_bbox.y + panel_bbox.height / 2

    dx = pcx - bubble_center_x
    dy = pcy - bubble_center_y

    # Determine primary direction
    v = ""
    if dy < -0.02:
        v = "up"
    elif dy > 0.02:
        v = "down"

    h = ""
    if dx < -0.02:
        h = "left"
    elif dx > 0.02:
        h = "right"

    if v and h:
        return f"{v}_{h}"
    if v:
        return v
    if h:
        return h
    return "down"  # fallback: most bubbles get a tail pointing down


# ---------------------------------------------------------------------------
# Text measurement  (CJK-aware)
# ---------------------------------------------------------------------------

def _load_font(size: int) -> "ImageFont.ImageFont":
    """Best-effort CJK font loading; falls back to PIL default."""
    import os as _os
    from PIL import ImageFont
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if _os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def estimate_text_size(
    text: str,
    font_size_pt: int = 12,
    max_width_px: int = 200,
) -> tuple[int, int]:
    """Measure the actual pixel size needed to render *text* using PIL.

    Loads the real CJK font at *font_size_pt*, measures each character's
    width with ``textbbox()``, and wraps lines at *max_width_px*.

    Returns (width_px, height_px) including padding.
    """
    if not text:
        return (80, 30)

_EMPTY_IMAGE = None


def _get_draw() -> "ImageDraw.ImageDraw":
    """Get a reusable PIL ImageDraw for text measurement (lazy init)."""
    global _EMPTY_IMAGE
    from PIL import Image, ImageDraw
    if _EMPTY_IMAGE is None:
        _EMPTY_IMAGE = Image.new("RGB", (1, 1))
    return ImageDraw.Draw(_EMPTY_IMAGE)


def estimate_text_size(
    text: str,
    font_size_pt: int = 12,
    max_width_px: int = 200,
) -> tuple[int, int]:
    """Measure the actual pixel size needed to render *text* using PIL.

    Loads the real CJK font at *font_size_pt*, measures each character's
    width with ``textbbox()``, and wraps lines at *max_width_px*.

    Returns (width_px, height_px) including padding.
    """
    if not text:
        return (80, 30)

    from PIL import ImageDraw

    font = _load_font(font_size_pt)
    draw = _get_draw()

    # Line breaking: accumulate characters until exceeding max_width_px
    lines: list[str] = []
    current_line = ""
    for ch in text:
        candidate = current_line + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        w = bbox[2] - bbox[0]
        if w > max_width_px and current_line:
            lines.append(current_line)
            current_line = ch
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)

    # Measure width of the longest rendered line
    max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        if w > max_w:
            max_w = w

    # Height: number of lines × line height
    # TrueType font .size returns the em height in pixels
    line_h = font.size + 4 if hasattr(font, "size") else font_size_pt + 4
    text_h = len(lines) * line_h

    # Add padding (matching compositor's _draw_bubble_text)
    pad = 10
    width = int(max_w) + pad * 2 + 4  # +4 for safety margin
    height = text_h + pad * 2

    # Clamp to reasonable limits
    width = max(80, min(width, 700))
    height = max(40, min(height, 500))

    return width, height


# ---------------------------------------------------------------------------
# Position suggestion  (from BubbleHint)
# ---------------------------------------------------------------------------

# Map BubbleHint position → relative anchor within the panel bbox
_POSITION_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left":      (0.02, 0.02),
    "top_right":     (0.55, 0.02),
    "bottom_left":   (0.02, 0.65),
    "bottom_right":  (0.55, 0.65),
    "center":        (0.25, 0.40),
    "auto":          (0.55, 0.02),   # default → top-right
}


def _bubble_anchor(panel_bbox: BoundingBox, hint: BubbleHint | None) -> tuple[float, float]:
    """Return the suggested (x, y) anchor within *panel_bbox* (normalised)."""
    if hint and hint.suggested_position:
        anchor = _POSITION_ANCHORS.get(hint.suggested_position, _POSITION_ANCHORS["auto"])
    else:
        anchor = _POSITION_ANCHORS["auto"]

    return (
        panel_bbox.x + panel_bbox.width * anchor[0],
        panel_bbox.y + panel_bbox.height * anchor[1],
    )


# ---------------------------------------------------------------------------
# Bubble-type resolution
# ---------------------------------------------------------------------------

def _resolve_bubble_type(dialogue: Dialogue, hint: BubbleHint | None) -> str:
    """Merge dialogue tone and BubbleHint to determine the bubble style."""
    # BubbleHint overrides if it specifies something beyond speech/thought
    if hint and hint.bubble_type not in ("speech", "thought"):
        return hint.bubble_type

    # Map Dialogue.tone → bubble type
    tone_map = {
        "angry":   "shout",
        "excited": "shout",
        "sad":     "whisper",
        "shy":     "whisper",
        "neutral": "speech",
    }
    tone = getattr(dialogue, "tone", "neutral") or "neutral"
    if dialogue.is_thought:
        return "thought"
    return tone_map.get(tone, "speech")


# ---------------------------------------------------------------------------
# Adaptive bubble size
# ---------------------------------------------------------------------------

# Shot-size → font-size multiplier (特写→更大，远景→更小)
_SHOT_SIZE_MULTIPLIER: dict[str, float] = {
    "extreme_long": 0.85,
    "long": 0.90,
    "full": 1.00,
    "medium": 1.05,
    "close": 1.10,
    "extreme_close": 1.15,
}


def calculate_bubble_size(
    text: str,
    panel_bbox: "BoundingBox",
    shot_size: str,
    emotion_intensity: float,
    page_width_px: int = 1240,
    page_height_px: int = 1754,
    margin_px: int = 40,
) -> tuple[float, float, int]:
    """根据面板大小、景别和情感强度计算椭圆气泡尺寸和字号。

    策略：按字号确定每行约 8 个 CJK 字 → 换行测量 → 按椭圆可用面积（75%）
    换算椭圆外框 → 约束宽高比（1.3:1 ~ 4.0:1）→ +5% 呼吸空间。
    """
    inner_w = page_width_px - margin_px * 2
    inner_h = page_height_px - margin_px * 2

    # 面板像素面积
    panel_w_px = panel_bbox.width * inner_w
    panel_h_px = panel_bbox.height * inner_h
    panel_area_px = panel_w_px * panel_h_px
    page_area_px = inner_w * inner_h

    # 基础字体：按面板占页面的比例缩放，18pt(min) ~ 34pt(max)
    area_ratio = panel_area_px / max(page_area_px, 1)
    base_font = 18 + int(area_ratio * 32)
    base_font = max(16, min(34, base_font))

    # 情感加成
    if emotion_intensity >= 0.85:
        base_font += 6
    elif emotion_intensity >= 0.70:
        base_font += 3

    # 景别调整
    shot_key = str(shot_size).lower() if hasattr(shot_size, "value") is False else shot_size
    multiplier = _SHOT_SIZE_MULTIPLIER.get(shot_key, 1.0)
    base_font = int(base_font * multiplier)
    base_font = max(14, min(38, base_font))

    # Step 1: 按字号确定文本换行宽度（每行约 8 个 CJK 字）
    text_max_w = int(base_font * 8)
    text_max_w = max(120, min(260, text_max_w))

    # Step 2: 换行测量
    text_w_px, text_h_px = estimate_text_size(text, base_font, max_width_px=text_max_w)

    # Step 3: 椭圆外框 = 文本 / 0.75（文本占椭圆 75%，inset ~12.5% 各边）
    ellipse_w = int(text_w_px / 0.75)
    ellipse_h = int(text_h_px / 0.75)

    # Step 4: 约束宽高比 1.3:1 ~ 4.0:1
    aspect = ellipse_w / max(ellipse_h, 1)
    if aspect > 4.0:
        ellipse_h = int(ellipse_w / 2.8)
    elif aspect < 1.3:
        ellipse_w = int(ellipse_h * 1.7)

    # Step 5: 呼吸空间 +5%
    ellipse_w = int(ellipse_w * 1.05)
    ellipse_h = int(ellipse_h * 1.05)

    # Step 6: 最小/最大约束
    ellipse_w = max(100, min(ellipse_w, inner_w - 20))
    ellipse_h = max(60, min(ellipse_h, inner_h - 20))

    # 归一化
    w_norm = ellipse_w / max(inner_w, 1)
    h_norm = ellipse_h / max(inner_h, 1)

    return w_norm, h_norm, base_font


# ---------------------------------------------------------------------------
# Smart bubble position
# ---------------------------------------------------------------------------

# 四角候选位 — 当有人物但未检测到人脸时使用，避免遮挡角色
_CORNER_CANDIDATES: list[tuple[str, float, float, str]] = [
    ("bottom_left",   0.02, 0.65, "up_right"),
    ("bottom_right",  0.55, 0.65, "up_left"),
    ("top_left",      0.02, 0.02, "down_right"),
    ("top_right",     0.55, 0.02, "down_left"),
]

# 分散候选位 — 当无人物（纯场景面板）时使用，均匀分布气泡
_DISPERSED_CANDIDATES: list[tuple[str, float, float, str]] = [
    ("top_left",      0.02, 0.02, "down_right"),
    ("top_right",     0.55, 0.02, "down_left"),
    ("top_center",    0.25, 0.02, "down"),
    ("bottom_left",   0.02, 0.65, "up_right"),
    ("bottom_right",  0.55, 0.65, "up_left"),
    ("bottom_center", 0.25, 0.65, "up"),
    ("mid_left",      0.02, 0.35, "right"),
    ("mid_right",     0.55, 0.35, "left"),
]

_GRID_SIZE: int = 5  # 5×5 = 25 个网格候选点


def _build_grid_candidates(
    panel_bbox: "BoundingBox",
    bubble_w: float,
    bubble_h: float,
) -> list[tuple[float, float]]:
    """在面板内生成 5×5 均匀网格的气泡中心候选点。

    网格点均匀分布在面板内部，为每个气泡留出尺寸边距，
    确保气泡不会超出面板边界。返回 (center_x, center_y) 列表。
    """
    margin_x = bubble_w / 2 + 0.02
    margin_y = bubble_h / 2 + 0.02

    x_min = panel_bbox.x + margin_x
    x_max = panel_bbox.x + panel_bbox.width - margin_x
    y_min = panel_bbox.y + margin_y
    y_max = panel_bbox.y + panel_bbox.height - margin_y

    # 钳制：防止气泡尺寸过大导致可用区域为负
    if x_min >= x_max:
        x_mid = panel_bbox.x + panel_bbox.width / 2
        x_min = x_mid - 0.01
        x_max = x_mid + 0.01
    if y_min >= y_max:
        y_mid = panel_bbox.y + panel_bbox.height / 2
        y_min = y_mid - 0.01
        y_max = y_mid + 0.01

    candidates: list[tuple[float, float]] = []
    for iy in range(_GRID_SIZE):
        y = y_min + (y_max - y_min) * iy / (_GRID_SIZE - 1)
        for ix in range(_GRID_SIZE):
            x = x_min + (x_max - x_min) * ix / (_GRID_SIZE - 1)
            candidates.append((x, y))
    return candidates


def _faces_to_page_coords(faces: list, panel_bbox: "BoundingBox") -> list[dict]:
    """将人脸坐标从面板图像空间 [0,1] 转换到页面全局空间。

    YOLO 返回的人脸坐标是相对于单个面板图像的归一化坐标，
    而气泡放置使用的是页面全局归一化坐标。此函数将人脸区域
    转换为页面空间，使重叠检测和距离计算在同一坐标系下进行。
    """
    if not faces:
        return []
    result: list[dict] = []
    for f in faces:
        fx = f.x if hasattr(f, 'x') else f.get("x", 0)
        fy = f.y if hasattr(f, 'y') else f.get("y", 0)
        fw = f.w if hasattr(f, 'w') else f.get("w", 0)
        fh = f.h if hasattr(f, 'h') else f.get("h", 0)
        result.append({
            "x": panel_bbox.x + fx * panel_bbox.width,
            "y": panel_bbox.y + fy * panel_bbox.height,
            "w": fw * panel_bbox.width,
            "h": fh * panel_bbox.height,
        })
    return result


def _bbox_overlap(
    bx: float, by: float, bw: float, bh: float,
    fx: float, fy: float, fw: float, fh: float,
) -> bool:
    """检测气泡 bbox 与人脸 bbox 是否重叠。"""
    return (bx < fx + fw and bx + bw > fx
            and by < fy + fh and by + bh > fy)


def _bbox_min_separation(
    bx: float, by: float, bw: float, bh: float,
    faces: list,
) -> float:
    """计算气泡 bbox 到最近人脸 bbox 的最小边到边距离。

    若气泡与任一人脸重叠 → 返回 0.0。
    若所有面都不重叠 → 返回最近边到边的欧氏距离。
    """
    if not faces:
        return float("inf")

    best = float("inf")
    for f in faces:
        fx = f.x if hasattr(f, 'x') else f.get("x", 0)
        fy = f.y if hasattr(f, 'y') else f.get("y", 0)
        fw = f.w if hasattr(f, 'w') else f.get("w", 0)
        fh = f.h if hasattr(f, 'h') else f.get("h", 0)

        h_overlap = bx < fx + fw and bx + bw > fx
        v_overlap = by < fy + fh and by + bh > fy

        if h_overlap and v_overlap:
            return 0.0  # 重叠 → 立即判死刑

        if h_overlap:
            # 仅垂直分离
            sep = min(abs(by - (fy + fh)), abs(fy - (by + bh)))
        elif v_overlap:
            # 仅水平分离
            sep = min(abs(bx - (fx + fw)), abs(fx - (bx + bw)))
        else:
            # 对角分离：两个轴上均不重叠
            dx = max(fx - (bx + bw), bx - (fx + fw), 0)
            dy = max(fy - (by + bh), by - (fy + fh), 0)
            sep = (dx * dx + dy * dy) ** 0.5

        if sep < best:
            best = sep

    return best


def _score_grid_candidate(
    bcx: float,
    bcy: float,
    bubble_w: float,
    bubble_h: float,
    faces: list,
    existing_bubbles: list["BubblePlacement"],
    panel_bbox: "BoundingBox",
) -> float:
    """对网格候选点计算综合评分（越高越优，≤0 表示不合格）。

    评分公式：
        0.50 × 人脸避让分（bbox 边到边距离，重叠直接判负）
        + 0.30 × 角落亲和力分（距最近面板角落的距离，越近越高）
        + 0.20 × 气泡间避让分

    若气泡 bbox 与任一人脸 bbox 重叠 → 返回 -1.0 直接淘汰。
    """
    score = 0.0

    # ---- 因子 1: 人脸避让 (权重 0.50，一票否决) ----
    bx = bcx - bubble_w / 2
    by = bcy - bubble_h / 2
    min_face_sep = _bbox_min_separation(bx, by, bubble_w, bubble_h, faces)
    if min_face_sep <= 0.0:
        return -1.0  # 与人脸重叠 → 直接淘汰
    # 甜点距离评分：最佳距离 0.08，太近(0)或太远(≥0.30)都降分
    # 避免气泡贴脸也避免气泡飞太远显得脱节
    sweet_spot = 0.08   # 最佳距离（页面宽度的 8%）
    far_limit = 0.30    # 超过此距离评分归零
    if min_face_sep <= sweet_spot:
        face_score = min_face_sep / sweet_spot           # 0 → 1.0
    else:
        face_score = max(0.0, 1.0 - (min_face_sep - sweet_spot) / (far_limit - sweet_spot))  # 1.0 → 0
    score += 0.50 * face_score

    # ---- 因子 2: 角落亲和力 (权重 0.30) ----
    # 漫画气泡最安全的位置是面板四角。计算气泡中心到最近角落的欧氏距离。
    corners = [
        (panel_bbox.x, panel_bbox.y),                                        # top-left
        (panel_bbox.x + panel_bbox.width, panel_bbox.y),                     # top-right
        (panel_bbox.x, panel_bbox.y + panel_bbox.height),                    # bottom-left
        (panel_bbox.x + panel_bbox.width, panel_bbox.y + panel_bbox.height), # bottom-right
    ]
    min_corner_dist = min(
        ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
        for cx, cy in corners
    )
    # 距离 0（紧贴角落）→ 1.0，距离 ≥ 面板对角线的 35% → 0
    corner_score = max(0.0, 1.0 - min_corner_dist / 0.35)
    score += 0.30 * corner_score

    # ---- 因子 3: 气泡间避让 (权重 0.20) ----
    if existing_bubbles:
        min_bubble_dist = float("inf")
        for eb in existing_bubbles:
            # 对已有气泡也用边到边距离
            eb_overlap = _bbox_overlap(
                bx, by, bubble_w, bubble_h,
                eb.x, eb.y, eb.w, eb.h,
            )
            if eb_overlap:
                # 与另一气泡重叠 → 降低评分
                if min_bubble_dist > 0.0:
                    min_bubble_dist = 0.0
            else:
                ecx = eb.x + eb.w / 2
                ecy = eb.y + eb.h / 2
                dist = ((bcx - ecx) ** 2 + (bcy - ecy) ** 2) ** 0.5
                if dist < min_bubble_dist:
                    min_bubble_dist = dist
        # 距离 ≥ 0.15 即满分
        bubble_score = min(min_bubble_dist / 0.15, 1.0)
    else:
        bubble_score = 1.0
    score += 0.20 * bubble_score

    return score


def smart_bubble_position(
    panel_bbox: "BoundingBox",
    faces: list,
    bubble_w: float,
    bubble_h: float,
    dialogue_index: int,
    num_dialogues: int,
    has_characters: bool = False,
    existing_bubbles: list["BubblePlacement"] | None = None,
) -> tuple[float, float, str]:
    """根据人脸/角色信息智能选择气泡位置和尾部方向。

    策略（优先级递减）：
    1. 检测到人脸 → 5×5 网格 + 多维度评分
    2. 有人物但无人脸 → 仅四角
    3. 无人物（纯场景）→ 8 个分散候选位循环
    """
    if existing_bubbles is None:
        existing_bubbles = []

    # ---- 策略 1: 有人脸 → 5×5 网格 + 多维评分 ----
    if faces:
        # 2a. 若人脸总面积超过面板 40%，网格无意义 → 直接退至四角
        panel_area = panel_bbox.width * panel_bbox.height
        total_face_area = sum(
            (f.w if hasattr(f, 'w') else f.get('w', 0))
            * (f.h if hasattr(f, 'h') else f.get('h', 0))
            for f in faces
        )
        face_area_ratio = total_face_area / panel_area if panel_area > 0 else 0.0

        if face_area_ratio > 0.40:
            # 人脸占据面板大部 — 回退到四角策略
            idx = dialogue_index % len(_CORNER_CANDIDATES)
            _label, ax_ratio, ay_ratio, tail = _CORNER_CANDIDATES[idx]
            bx = panel_bbox.x + panel_bbox.width * ax_ratio
            by = panel_bbox.y + panel_bbox.height * ay_ratio
            # 宽松钳制：允许气泡出框
            bx = max(-bubble_w * 0.5, min(bx, 1.0 - bubble_w * 0.5))
            by = max(-bubble_h * 0.5, min(by, 1.0 - bubble_h * 0.5))
            return bx, by, tail

        # 2b. 正常情况 — 5×5 网格评分
        grid = _build_grid_candidates(panel_bbox, bubble_w, bubble_h)
        best_score = -1.0
        best_bx = best_by = 0.0
        for cx, cy in grid:
            score = _score_grid_candidate(
                cx, cy, bubble_w, bubble_h, faces,
                existing_bubbles, panel_bbox,
            )
            if score > best_score:
                best_score = score
                best_bx = cx - bubble_w / 2
                best_by = cy - bubble_h / 2

        if best_score <= 0.0:
            # 全部网格点都与人脸重叠 → 退至四角策略
            idx = dialogue_index % len(_CORNER_CANDIDATES)
            _label, ax_ratio, ay_ratio, tail = _CORNER_CANDIDATES[idx]
            best_bx = panel_bbox.x + panel_bbox.width * ax_ratio
            best_by = panel_bbox.y + panel_bbox.height * ay_ratio

        # 宽松钳制：仅防气泡完全跑出页面，不阻止合理的面板边缘出框
        best_bx = max(-bubble_w * 0.5, min(best_bx, 1.0 - bubble_w * 0.5))
        best_by = max(-bubble_h * 0.5, min(best_by, 1.0 - bubble_h * 0.5))

        best_tail = _derive_tail_direction(
            best_bx + bubble_w / 2, best_by + bubble_h / 2, panel_bbox,
        )
        return best_bx, best_by, best_tail

    # ---- 策略 3 & 4: 无人脸 → 根据角色存在选择候选集 ----
    if has_characters:
        # 有人物但未检测到人脸 → 仅四角，避免遮挡角色
        candidates = _CORNER_CANDIDATES
    else:
        # 无人物（纯场景面板）→ 8 个分散候选位
        candidates = _DISPERSED_CANDIDATES

    idx = dialogue_index % len(candidates)
    _label, ax_ratio, ay_ratio, tail = candidates[idx]
    bx = panel_bbox.x + panel_bbox.width * ax_ratio
    by = panel_bbox.y + panel_bbox.height * ay_ratio
    # 宽松钳制：允许气泡出框
    bx = max(-bubble_w * 0.5, min(bx, 1.0 - bubble_w * 0.5))
    by = max(-bubble_h * 0.5, min(by, 1.0 - bubble_h * 0.5))
    return bx, by, tail


# ---------------------------------------------------------------------------
# Bubble style parameters
# ---------------------------------------------------------------------------

def bubble_style_params(bubble_type: str, bubble_area_px: float) -> dict:
    """根据气泡类型和面积返回样式参数。

    面积越大 → 边框和圆角越大。所有类型统一白底黑框，
    仅 border/radius/tail 根据面积和类型微调。
    """
    # 面积越大 → 边框和圆角越大
    if bubble_area_px > 80000:
        border_w, radius, tail = 3, 16, 16
    elif bubble_area_px > 30000:
        border_w, radius, tail = 2, 12, 12
    else:
        border_w, radius, tail = 1, 8, 8

    # 统一白底黑框，仅 border/radius/tail 按类型微调
    base = {
        "fill": [255, 255, 255],
        "outline": [20, 20, 20],
        "border": border_w,
        "radius": radius,
        "tail": tail,
    }
    if bubble_type == "shout":
        base["border"] = max(2, border_w)
        base["radius"] = 6
    elif bubble_type == "narration":
        base["radius"] = 4
        base["tail"] = 0

    return base


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def place_bubbles(
    panels: list[PanelPlan],
    panel_bboxes: list[BoundingBox],
    panel_images: dict[str, PanelImage],
    *,
    font_size_pt: int = 12,
    page_width_px: int = 1240,
    page_height_px: int = 1754,
    margin_px: int = 40,
    faces_by_panel: dict[str, list] | None = None,
) -> list[BubblePlacement]:
    """Compute bubble positions for all panels on a page.

    Parameters
    ----------
    panels:
        Panel plans in reading order (from StoryboardAgent).
    panel_bboxes:
        Solved bboxes, parallel to *panels*.
    panel_images:
        Generated panel images keyed by panel_id (for future image analysis).
    font_size_pt:
        Base font size for text-size estimation.
    faces_by_panel:
        Optional dict of panel_id → list[FaceRegion] from VLM detection.
        When provided, bubble positions are optimized to avoid occluding faces.

    Returns
    -------
    list[BubblePlacement]
        All bubbles for the page, with normalised coordinates.
    """
    if faces_by_panel is None:
        faces_by_panel = {}

    inner_w = page_width_px - margin_px * 2
    inner_h = page_height_px - margin_px * 2
    use_adaptive = font_size_pt <= 0  # <=0 表示启用自适应尺寸

    all_bubbles: list[BubblePlacement] = []

    # 旁白位置：紧贴四个角落
    _NARRATION_CORNERS: list[tuple[float, float]] = [
        (0.02, 0.84),   # bottom_left  — 紧贴左下角
        (0.60, 0.84),   # bottom_right — 紧贴右下角
        (0.02, 0.02),   # top_left    — 紧贴左上角
        (0.60, 0.02),   # top_right   — 紧贴右上角
    ]

    for panel, bbox in zip(panels, panel_bboxes, strict=False):
        dialogues = panel.dialogues_in_panel
        narration_text = getattr(panel, "narration", "") or ""

        if not dialogues and not narration_text:
            continue

        # 该面板的人脸检测结果 → 转换到页面全局坐标系
        panel_faces = faces_by_panel.get(panel.panel_id, [])
        page_faces = _faces_to_page_coords(panel_faces, bbox)

        # Build hint lookup keyed by dialogue_index
        hint_map: dict[int, BubbleHint] = {}
        for h in panel.speech_bubble_hints:
            hint_map[h.dialogue_index] = h

        # 获取景别和情感（用于自适应尺寸）
        shot_size = (
            panel.shot_size.value
            if hasattr(panel.shot_size, "value")
            else str(panel.shot_size)
        )
        emotion = getattr(panel, "emotion_intensity", 0.5) or 0.5

        num_dialogues = len(dialogues)

        # 判断面板是否有人物角色（决定无人脸时的候选位置策略）
        has_characters = bool(getattr(panel, "characters_in_panel", None))

        # Place each dialogue
        for di, dialogue in enumerate(dialogues):
            hint = hint_map.get(di)

            # Resolve bubble type
            bubble_type = _resolve_bubble_type(dialogue, hint)

            # ---- 尺寸计算 ----
            if use_adaptive:
                w_norm, h_norm, bubble_font_pt = calculate_bubble_size(
                    dialogue.text, bbox, shot_size, emotion,
                    page_width_px, page_height_px, margin_px,
                )
            else:
                text_w_px, text_h_px = estimate_text_size(dialogue.text, font_size_pt)
                w_norm = text_w_px / max(inner_w, 1)
                h_norm = text_h_px / max(inner_h, 1)
                bubble_font_pt = font_size_pt

            # ---- 智能位置选择 ----
            # 收集当前面板已有气泡（用于气泡间避让）
            panel_existing = [b for b in all_bubbles if b.panel_id == panel.panel_id]
            bx, by, tail_dir = smart_bubble_position(
                bbox, page_faces,
                w_norm, h_norm,
                dialogue_index=di,
                num_dialogues=num_dialogues,
                has_characters=has_characters,
                existing_bubbles=panel_existing,
            )

            # ---- 样式参数 ----
            bubble_area_px = (w_norm * inner_w) * (h_norm * inner_h)
            style = bubble_style_params(bubble_type, bubble_area_px)

            all_bubbles.append(
                BubblePlacement(
                    panel_id=panel.panel_id,
                    dialogue_index=di,
                    bubble_type=bubble_type,
                    speaker=dialogue.speaker,
                    text=dialogue.text,
                    x=bx,
                    y=by,
                    w=w_norm,
                    h=h_norm,
                    font_size_pt=bubble_font_pt,
                    occlusion_score=0.0,
                    tail_direction=tail_dir,
                    style=style,
                )
            )

        # ---- 旁白处理 ----
        if narration_text:
            # 旁白用固定适中的字号，不参与自适应
            narr_w_px, narr_h_px = estimate_text_size(
                narration_text, font_size_pt=16, max_width_px=300,
            )
            # 旁白方框尺寸 = 文本 + 内边距
            narr_w_px = int(narr_w_px * 1.2)
            narr_h_px = int(narr_h_px * 1.2)
            w_norm = narr_w_px / max(inner_w, 1)
            h_norm = narr_h_px / max(inner_h, 1)

            # 放置到不受气泡干扰的角落（优先 bottom_left）
            best_corner = _NARRATION_CORNERS[0]
            for cx_ratio, cy_ratio in _NARRATION_CORNERS:
                cx = bbox.x + bbox.width * cx_ratio
                cy = bbox.y + bbox.height * cy_ratio
                # 钳制在面板内
                cx = max(bbox.x + 0.01, min(cx, bbox.x + bbox.width - w_norm - 0.01))
                cy = max(bbox.y + 0.01, min(cy, bbox.y + bbox.height - h_norm - 0.01))
                # 避开已有对话气泡
                for existing in all_bubbles:
                    if existing.panel_id != panel.panel_id:
                        continue
                    # 简单重叠检测
                    overlap_x = abs(cx - existing.x) < (w_norm + existing.w) * 0.6
                    overlap_y = abs(cy - existing.y) < (h_norm + existing.h) * 0.6
                    if overlap_x and overlap_y:
                        break
                else:
                    best_corner = (cx, cy)
                    break

            narr_area_px = (w_norm * inner_w) * (h_norm * inner_h)
            style = bubble_style_params("narration", narr_area_px)

            all_bubbles.append(
                BubblePlacement(
                    panel_id=panel.panel_id,
                    dialogue_index=num_dialogues,  # 排在对话之后
                    bubble_type="narration",
                    speaker="",
                    text=narration_text,
                    x=best_corner[0],
                    y=best_corner[1],
                    w=w_norm,
                    h=h_norm,
                    font_size_pt=16,
                    occlusion_score=0.0,
                    tail_direction="auto",
                    style=style,
                )
            )

    return all_bubbles
