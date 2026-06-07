"""YOLO-based face detection — 替代 VLM API 方案，纯本地、极快。

使用 face_yolov8n.pt 检测黑白漫画面板中的人脸，
返回归一化的 FaceRegion 列表。

与 face_detector.py 的关系：
- face_detector.py 保留不动（含 FaceRegion、optimize_bubble_positions、detect_faces_vlm）
- 本模块是新增的 YOLO 后端，输出同格式的 FaceRegion
- optimize_bubble_positions() 完全复用，不需要改动
"""

import asyncio
import logging
from pathlib import Path

from .face_detector import FaceRegion

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME = "face_yolov8n.pt"


def _get_model():
    """懒加载 YOLO 模型（全局单例）。"""
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO

        # 模型路径：项目根目录 yolo/
        model_path = (
            Path(__file__).parent.parent.parent.parent.parent / "yolo" / _MODEL_NAME
        )
        logger.info("加载 YOLO 人脸检测模型: %s", model_path)
        _MODEL = YOLO(str(model_path))
    return _MODEL


async def detect_faces_yolo(image_path: str) -> list[FaceRegion]:
    """使用 YOLO 检测人脸，返回归一化 FaceRegion 列表。

    参数
    ----------
    image_path : str
        面板图像的本地路径。

    返回
    -------
    list[FaceRegion]
        检测到的人脸区域，坐标为归一化 [0,1] 值。
        无人脸或检测失败时返回空列表。
    """
    try:
        model = _get_model()
        results = await asyncio.to_thread(model, str(image_path), verbose=False)

        faces: list[FaceRegion] = []
        for r in results:
            if r.boxes is None:
                continue
            orig_h, orig_w = r.orig_shape[:2]
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if conf < 0.3:  # 低置信度过滤
                    continue
                faces.append(
                    FaceRegion(
                        x=x1 / orig_w,  # 归一化到 0-1
                        y=y1 / orig_h,
                        w=(x2 - x1) / orig_w,
                        h=(y2 - y1) / orig_h,
                    )
                )
        return faces

    except FileNotFoundError:
        logger.warning("YOLO 模型文件未找到，跳过人脸检测")
        return []
    except Exception as exc:
        logger.warning("YOLO 人脸检测失败 (%s)，回退到纯启发式放置", exc)
        return []
