"""YOLO-based face detection — 替代 VLM API 方案，纯本地、极快。

使用 face_yolov8n.pt 检测黑白漫画面板中的人脸，
返回归一化的 FaceRegion 列表。

模型路径通过 configs/comicweaver.yaml 的 yolo.model_path 配置。

与 face_detector.py 的关系：
- face_detector.py 保留不动（含 FaceRegion、optimize_bubble_positions、detect_faces_vlm）
- 本模块是新增的 YOLO 后端，输出同格式的 FaceRegion
- optimize_bubble_positions() 完全复用，不需要改动
"""

import asyncio
import logging
import threading
from pathlib import Path

from PIL import Image

from .face_detector import FaceRegion

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_PATH = None
_MODEL_MISSING_WARNED = False  # 全局标记，避免重复警告
_MODEL_LOCK = threading.Lock()  # ★ 线程锁 — YOLO 模型不支持并发推理


def _get_model(model_path: str):
    """懒加载 YOLO 模型（按路径单例）。"""
    global _MODEL, _MODEL_PATH, _MODEL_MISSING_WARNED
    if _MODEL is None or _MODEL_PATH != model_path:
        if not Path(model_path).exists():
            if not _MODEL_MISSING_WARNED:
                logger.warning(
                    "⚠️ YOLO 人脸检测模型文件不存在: %s\n"
                    "   人脸检测将不可用，气泡将使用纯启发式放置。\n"
                    "   请确保 yolo/face_yolov8n.pt 文件存在。",
                    model_path,
                )
                _MODEL_MISSING_WARNED = True
            return None

        from ultralytics import YOLO

        _MODEL = YOLO(str(model_path))
        _MODEL_PATH = model_path
        logger.info("✅ 加载 YOLO 人脸检测模型: %s", model_path)
    return _MODEL


async def detect_faces_yolo(
    image_path: str,
    model_path: str = "yolo/face_yolov8n.pt",
    confidence_threshold: float = 0.15,
) -> list[FaceRegion]:
    """使用 YOLO 检测人脸，返回归一化 FaceRegion 列表。

    参数
    ----------
    image_path : str
        面板图像的本地路径。
    model_path : str
        YOLO 模型文件路径（相对于项目根目录或绝对路径）。
    confidence_threshold : float
        最低置信度阈值（0.0~1.0），低于此值的检测结果被丢弃。
        默认 0.15，针对漫画/动漫风格人脸调低阈值（照片人脸通常用 0.3+）。

    返回
    -------
    list[FaceRegion]
        检测到的人脸区域，坐标为归一化 [0,1] 值。
        无人脸或检测失败时返回空列表。
    """
    try:
        # 解析模型路径：相对路径 → 相对于项目根目录
        model_p = Path(model_path)
        if not model_p.is_absolute():
            model_p = (
                Path(__file__).parent.parent.parent.parent.parent / model_path
            )

        model = _get_model(str(model_p))
        if model is None:
            return []  # 模型文件不存在，已在 _get_model 中警告

        # 预处理：确保图像为 RGB 格式（YOLO 在 RGB 上训练，RGBA/灰度可能降低检测率）
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            # 保存临时 RGB 副本供 YOLO 使用
            tmp_path = Path(image_path).with_suffix(".rgb_temp.png")
            img.save(tmp_path)
            detect_path = str(tmp_path)
        else:
            detect_path = str(image_path)

        # YOLO 推理 — 必须加锁，模型不支持多线程并发调用
        # 多线程同时调用 model() 会导致 'Conv' object has no attribute 'bn' 等错误
        with _MODEL_LOCK:
            results = await asyncio.to_thread(model, detect_path, verbose=False)

        # 清理临时文件
        if detect_path != str(image_path):
            try:
                Path(detect_path).unlink()
            except OSError:
                pass

        faces: list[FaceRegion] = []
        for r in results:
            if r.boxes is None:
                continue
            orig_h, orig_w = r.orig_shape[:2]
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                faces.append(
                    FaceRegion(
                        x=x1 / orig_w,
                        y=y1 / orig_h,
                        w=(x2 - x1) / orig_w,
                        h=(y2 - y1) / orig_h,
                    )
                )

        if not faces and len(results) > 0 and results[0].boxes is not None:
            # 有检测结果但都被阈值过滤了 — 记录低置信度检测
            low_conf_count = len(results[0].boxes)
            best_conf = max(float(b.conf[0]) for b in results[0].boxes) if results[0].boxes else 0
            logger.debug(
                "面板 %s: %d 个检测被阈值 %.2f 过滤（最高置信度: %.3f），"
                "考虑降低 confidence_threshold",
                Path(image_path).stem, low_conf_count, confidence_threshold, best_conf,
            )

        return faces

    except FileNotFoundError:
        logger.warning(
            "⚠️ YOLO 模型文件未找到: %s，跳过人脸检测。"
            "气泡将使用纯启发式放置（不考虑人脸遮挡）。",
            model_path,
        )
        return []
    except Exception as exc:
        logger.warning(
            "⚠️ YOLO 人脸检测失败 (%s)，回退到纯启发式放置。"
            "这可能导致气泡遮挡人物面部。",
            exc,
        )
        return []
