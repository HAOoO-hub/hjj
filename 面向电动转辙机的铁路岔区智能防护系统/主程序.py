import os
import time
import json
import math
import platform
import subprocess
import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# 按 connection.py 的方式：FFmpeg 超时 5000ms
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"

import cv2
import numpy as np
import requests
from PIL import Image

from datetime import datetime
from yolo import YOLO, YOLO_ONNX
import torch

'''

状态总共有三个值
0：表示没有物体出现在目标铁轨
1：疑似有物品靠近目标铁轨
2：表示有火车在目标铁轨出现
3：表示有人在目标铁轨出现

'''

CONFIG: Dict[str, Any] = {
    # ==========================================================
    # 1) 运行模式
    # ==========================================================
    # "camera": RTSP 摄像头模式
    # "video" : 本地视频文件模式
    "input_mode": "video",
    "show_window": True,
    "save_output_video": True,

    # 摄像头抓流线程参数（仅 camera 模式使用）
    "grab_interval": 0.05,
    "frame_stale_seconds": 2.0,
    "post_reconnect_grace_seconds": 1.5,
    "cooldown_network_seconds": 60,
    "cooldown_service_seconds": 30,

    # 0 表示每帧都检测；1 表示隔 1 帧检测一次；2 表示隔 2 帧检测一次...
    "detect_interval_frames": 0,
    # 连续 3 次相同的采样结果，才更新为稳定状态
    "stable_required_samples": 2,

    # ==========================================================
    # 2) 模型配置
    # ==========================================================
    "backend": "pytorch",  # "pytorch" or "onnx"
    # "backend": "onnx",  # "pytorch" or "onnx"
    "model_path": r"model_data/l_train_person.pth",
    # "model_path": r"model_data/l_train_person.onnx",
    "classes_path": r"model_data/cls_classes_train_person.txt",
    "phi": "l",
    "cuda":True,

    # ==========================================================
    # 3) 输入输出配置
    # ==========================================================
    "camera": {
        "name": "target_rail_camera",
        "device_id": 1,#设备编号
        # "rtsp_url": "rtsp://admin:aabbcc666@192.168.1.64:554/h264/ch2/sub/av_stream",#摄像头rstp地址
        "rtsp_url": "rtsp://127.0.0.1:8554/z4_train",#摄像头rstp地址
        "output_video_path": r"output/camera_result.mp4",
    },
    "video": {
        # "video_path": r"E:\video\Z4\2026-03-30\Z4_train.mp4",
        # "video_path": r"E:\video\Z4\2026-03-30\Z4_train_useful.mp4",
        "video_path": r"E:\video\output\5.mp4",
        # "video_path": r"E:\video\output\12\5\5_part_of.mp4",
        # "video_path": r"E:\video\output\31.mp4",
        # "video_path": r"E:\video\Z4\2026-03-30\person_cut\person_cut.mp4",
        # "video_path": r"E:\video\Z4\2026-03-30\train_pass\train_pass.mp4",
        # "video_path": r"D:\video\train_and_person\Z2\person_cut\person_cut.mp4",
        # "video_path": r"D:\video\train_and_person\Z2\train_cut\train_cut1\train_cut1.mp4",
        # "video_path": r"D:\video\train_and_person\Z2\train_cut\train_cut2\train_cut2_1\train_cut2_1.mp4",
        # "video_path": r"D:\video\train_and_person\Z2\train_cut\train_cut2\train_cut2_2\2\2.mp4",
        # "video_path": r"D:\video\train_and_person\Z2\train_cut\train_cut2\train_cut2_2\train_cut2_2.mp4",
        # "output_video_path": r"output/Z4_train_pass_result.mp4",
        # "output_video_path": r"output/Z4_train_pass_result1.mp4",
        # "output_video_path": r"output/Z4_train_result.mp4",
        # "output_video_path": r"output/Z4_train_useful_result.mp4",
        # "output_video_path": r"output/Z2_train_cut1_result.mp4",
        # "output_video_path": r"output/Z2_train_cut2_1_result.mp4",
        # "output_video_path": r"output/Z2_2_result.mp4",
        # "output_video_path": r"output/Z2_train_cut2_2_result.mp4",
        "output_video_path": r"output/5.mp4",
        # "output_video_path": r"output/person_cut.mp4",
    },

    # ==========================================================
    # 4) 多边形区域
    # ==========================================================
    # 目标铁轨区域：用于 person 靠近/进入、train 进入目标轨道、canny ROI
    "target_rail_polygon": [
        #Z4目标铁轨区域
        (924, 283), (1440, 1229), (2276, 1034), (1877, 761),
        (1600, 576), (1419, 459), (1277, 370), (1192, 301), (1163, 262)

        #Z2目标铁轨区域
        # (930, 1352), (1250, 622), (1404, 299), (1440, 212),
        # (1211, 208), (1178, 255), (992, 405), (599, 703), (91, 1159)
    ],

    # 掩码区域：第一遍检测前将这些区域涂黑
    # 支持 1 个或多个 polygon
    "mask_polygons": [
        [

            #Z4掩码区域，掩码区域临近右侧铁轨
            # (2549, 618), (1606, 310), (1184, 220), (1078, 108), (863, 97), (830, 247), (578, 707), (239, 1433), (2555, 1433)

            # Z4掩码区域调整
            # (1240, 226), (1096, 2), (878, 4), (832, 276), (249, 1415), (2551, 1431), (2553, 597), (1777, 347), (1388, 264)

            # (2549, 580), (1839, 347), (1558, 283), (1392, 251), (1321, 239), (1236, 178), (1125, 91), (1034, 4), (857, 6), (820, 251), (495, 849), (289, 1429), (2551, 1433)
            #完全掩盖
            (2, 4), (2, 1435), (2551, 1433), (2551, 4)
            #扩大的掩码区域，靠近右侧第二轨道
            # (2553, 509), (2010, 355), (1392, 235), (1242, 218), (1109, 2), (872, 0), (830, 243), (570, 707), (251, 1429), (2547, 1431)

            #Z2掩码区域，掩码区域靠近左侧铁轨
            # (2, 865), (718, 395), (974, 255), (1130, 178), (1219, 2), (1434, 0), (1471, 285), (1465, 1433), (2, 1433)

        ]
    ],

    # ==========================================================
    # 5) 类别与阈值
    # ==========================================================
    "person_labels": ["person"],
    "train_labels": ["train"],

    # 模型原始框过滤阈值
    "default_conf_threshold": 0.35,
    "class_conf_thresholds": {
        "person": 0.70,
        "train": 0.35,
    },

    # 双遍检测：第二遍结果中删除第一遍中的 train 框
    "same_train_overlap_threshold": 0.6,

    # 小框被大框包含的删除条件
    # 条件1：intersection / area_small > 0.75
    # 条件2：中心点接近且中心点位于大框内部
    "nested_box_contain_threshold": 0.75,
    "nested_center_distance_ratio": 0.50,
    # "nested_min_center_distance_px": 12.0,
    "nested_min_center_distance_px": 12.0,

    # ==========================================================
    # 6) 行人与铁轨的关系判定
    # ==========================================================
    "person_near_threshold": 40,
    "person_foot_inset_ratio": 0.15,

    # ==========================================================
    # 7) train 目标轨道区域判定
    # overlap_area > 1844  -> 有火车正在驶过
    # 454 <= overlap_area <= 1844 -> 火车开始进入/预警
    # < 454 -> 无
    # ==========================================================
    "train_overlap_warning_area": 454,
    "train_overlap_train_area": 1844,

    # ==========================================================
    # 8) Canny / 模板配置
    # r_keep < 0.77   -> train
    # 0.77~0.87       -> warning
    # >= 0.87         -> none
    # ==========================================================
    "edge_reference_path": r"template_update_output/updated_stable_edge_template.png",
    "canny_low": 50,
    "canny_high": 150,
    "gaussian_ksize": (5, 5),
    "gaussian_sigma": 1.0,
    "track_keep_low_th": 0.75,
    "track_keep_high_th": 0.85,

    # 每 5 分钟尝试更新一次模板；开始更新前需确认目标轨道区域无 train/person
    "template_update_enabled": True,
    "template_update_interval_seconds": 300,
    # 当 box 判到目标轨道有车时，是否要求 Canny 模板也判到“有车/变化”才最终确认有车
    "train_template_confirm_enabled": True,
    # True: 只要模板结果不是 CLEAR（WARNING/TRAIN）就算“模板也判到有车”
    # False: 只有模板结果为 TRAIN 才算“模板也判到有车”
    "train_template_confirmation_accept_warning": True,
    "template_update_take_every_n_frames": 5,
    "template_update_need_frames": 20,
    "template_output_dir": "template_update_output",
    "template_keep_ratio": 0.70,
    "template_bootstrap_on_start": True,
    "template_update_old_weight": 0.7,
    "template_update_new_weight": 0.3,
    "template_update_diff_threshold": 0.15,
    "template_update_binarize_threshold": 0.5,
    "template_update_edge_ratio_low": 0.7,
    "template_update_edge_ratio_high": 1.3,
    "template_library_max_size": 3,
    "template_library_similarity_diff_threshold": 0.10,
    "template_library_similarity_edge_ratio_low": 0.85,
    "template_library_similarity_edge_ratio_high": 1.15,

    # 坏模板恢复：检测框判定轨道干净，但 Canny 连续误报时，触发模板重建
    "bad_template_recovery_enabled": True,
    "bad_template_suspect_threshold": 30,
    "bad_template_force_rebuild_on_trigger": True,
    "bad_template_bypass_update_interval_on_trigger": True,

    # ==========================================================
    # 9) 接口上报配置
    # 0=clear, 1=warning, 2=train, 3=person
    # 仅在稳定状态发生变化时上报
    # ==========================================================
    "api_push_enabled": True,
    "heartbeat_enabled": True,
    "heartbeat_init_status": -1,
    "heartbeat_interval_seconds": 300,
    "external_api": {
        "base_url": "http://test.agilefast.cn:8087/SmartAI",
        # "base_url": "http://127.0.0.1:8080",
        "endpoint": "/SmartAI/ai/objectdetection/pushRailStatus",
        "device_id_param": "deviceId",
        "status_param": "railStatus",
        "extra_params": {},
        "timeout": 5,
    },

    # ==========================================================
    # 10) 日志配置
    # ==========================================================
    "log_enabled": True,
    "log_level": "INFO",
    "log_file_path": "logs/railway_monitor.log",

    # ==========================================================
    # 11) 绘图配置
    # ==========================================================
    "draw_target_polygon": True,
    "draw_mask_polygons": True,
    "draw_person_boxes": True,
    "draw_train_boxes": True,
    "draw_removed_trains": False,
    "draw_person_points": True,
}


STATE_CLEAR = 0
STATE_WARNING = 1
STATE_TRAIN = 2
STATE_PERSON = 3

CONNECTION_ONLINE = "online"
CONNECTION_OFFLINE = "offline"
CONNECTION_STARTING = "starting"


# ============================================================
# 日志
# ============================================================
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("railway_monitor")
    level_name = str(CONFIG.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if bool(CONFIG.get("log_enabled", True)):
        log_file_path = str(CONFIG.get("log_file_path", "logs/railway_monitor.log"))
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = setup_logger()


def log_info(message: str) -> None:
    LOGGER.info(message)


def log_warning(message: str) -> None:
    LOGGER.warning(message)


def log_error(message: str) -> None:
    LOGGER.error(message)


# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def to_polygon(points: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32)


def polygons_from_config(raw_polygons: Sequence[Sequence[Sequence[int]]]) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    for item in raw_polygons:
        if len(item) >= 3:
            polys.append(to_polygon(item))
    return polys


def open_video_capture(video_path: str):
    backends = []
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for backend in backends:
        try:
            cap = cv2.VideoCapture(video_path) if backend is None else cv2.VideoCapture(video_path, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return None


def create_video_writer(output_path: str, fps: float, width: int, height: int):
    ensure_dir(os.path.dirname(output_path))
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def tlbr_to_xyxy(box_tlbr: Sequence[float]) -> Tuple[float, float, float, float]:
    top, left, bottom, right = box_tlbr
    return float(left), float(top), float(right), float(bottom)


def normalize_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    return x1, y1, x2, y2


def box_area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def iou_xyxy(box1: Sequence[float], box2: Sequence[float]) -> float:
    inter = intersection_area_xyxy(box1, box2)
    union = box_area_xyxy(box1) + box_area_xyxy(box2) - inter
    return inter / (union + 1e-6) if union > 0 else 0.0


def overlap_ratio_on_small(box_small: Sequence[float], box_big: Sequence[float]) -> float:
    inter = intersection_area_xyxy(box_small, box_big)
    area_small = box_area_xyxy(box_small)
    return inter / (area_small + 1e-6) if area_small > 0 else 0.0


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def point_in_box(pt: Tuple[float, float], box: Sequence[float]) -> bool:
    px, py = pt
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def center_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    c1 = box_center(box1)
    c2 = box_center(box2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def rect_corners(box: Sequence[float]) -> List[Tuple[float, float]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def point_in_rect(pt: Sequence[float], box: Sequence[float]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def orientation(a, b, c) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-6:
        return 0
    return 1 if val > 0 else 2


def on_segment(a, b, c) -> bool:
    return (
        min(a[0], c[0]) - 1e-6 <= b[0] <= max(a[0], c[0]) + 1e-6 and
        min(a[1], c[1]) - 1e-6 <= b[1] <= max(a[1], c[1]) + 1e-6
    )


def segments_intersect(p1, q1, p2, q2) -> bool:
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def rect_intersects_polygon(box: Sequence[float], polygon: np.ndarray) -> bool:
    for pt in rect_corners(box):
        if cv2.pointPolygonTest(polygon, pt, False) >= 0:
            return True
    for pt in polygon.tolist():
        if point_in_rect(pt, box):
            return True
    rc = rect_corners(box)
    rect_edges = list(zip(rc, rc[1:] + rc[:1]))
    poly_pts = [tuple(map(float, p)) for p in polygon.tolist()]
    poly_edges = list(zip(poly_pts, poly_pts[1:] + poly_pts[:1]))
    for e1 in rect_edges:
        for e2 in poly_edges:
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False


def apply_black_mask(frame: np.ndarray, polygons: Sequence[np.ndarray]) -> np.ndarray:
    masked = frame.copy()
    for poly in polygons:
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(masked, [poly.reshape((-1, 1, 2))], (0, 0, 0))
    return masked


# ============================================================
# 模型加载与检测
# ============================================================
def load_detector() -> object:
    backend = str(CONFIG["backend"]).lower()
    model_path = str(CONFIG["model_path"])
    classes_path = str(CONFIG["classes_path"])

    if backend == "onnx":
        return YOLO_ONNX(onnx_path=model_path, classes_path=classes_path)

    if backend == "pytorch":
        # use_cuda = torch.cuda.is_available()
        use_cuda = CONFIG["cuda"]
        log_info(f"[detector] pytorch use_cuda={use_cuda}")
        return YOLO(
            model_path=model_path,
            classes_path=classes_path,
            phi=str(CONFIG.get("phi", "s")),
            cuda=use_cuda,
        )

    raise ValueError(f"不支持的 backend: {backend}")


def get_conf_threshold(label_name: str) -> float:
    label_name = label_name.lower()
    class_thresholds = CONFIG.get("class_conf_thresholds", {})
    if isinstance(class_thresholds, dict) and label_name in class_thresholds:
        return float(class_thresholds[label_name])
    return float(CONFIG["default_conf_threshold"])


def filter_by_labels(detections: List[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    label_set = {x.lower() for x in labels}
    return [d for d in detections if d["label_lower"] in label_set]


def remove_nested_duplicates(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(detections) <= 1:
        return detections

    contain_thresh = float(CONFIG["nested_box_contain_threshold"])
    center_dist_ratio = float(CONFIG.get("nested_center_distance_ratio", 0.5))
    min_center_px = float(CONFIG.get("nested_min_center_distance_px", 12.0))

    detections_sorted = sorted(
        detections,
        key=lambda d: (d["label_lower"], -box_area_xyxy(d["box_xyxy"]), -float(d["confidence"])),
    )

    kept: List[Dict[str, Any]] = []
    for det in detections_sorted:
        cur_box = det["box_xyxy"]
        cur_area = box_area_xyxy(cur_box)
        if cur_area <= 0:
            continue

        should_drop = False
        for kept_det in kept:
            if det["label_lower"] != kept_det["label_lower"]:
                continue

            big_box = kept_det["box_xyxy"]
            inter_ratio = intersection_area_xyxy(cur_box, big_box) / (cur_area + 1e-6)
            center_inside = point_in_box(box_center(cur_box), big_box)

            x1, y1, x2, y2 = cur_box
            sw = max(1.0, x2 - x1)
            sh = max(1.0, y2 - y1)
            allowed_center_dist = max(min_center_px, min(sw, sh) * center_dist_ratio)
            center_close = center_distance(cur_box, big_box) <= allowed_center_dist

            if det["label_lower"] == "train":
                if inter_ratio >= 0.70 and center_inside:
                    should_drop = True
                    break
            else:
                if inter_ratio >= contain_thresh and center_inside and center_close:
                    should_drop = True
                    break

        if not should_drop:
            kept.append(det)

    return kept


def detect_frame_raw(detector: object, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t1 = time.time()
    raw = detector.detect_image(image_pil)
    t2 = time.time()
    print(f"detect one frame need {t2-t1}")

    if raw is None or not isinstance(raw, tuple) or len(raw) != 3:
        return []

    boxes, labels, confs = raw
    h, w = frame_bgr.shape[:2]
    detections: List[Dict[str, Any]] = []

    for box_tlbr, label_id, conf in zip(boxes, labels, confs):
        label_name = detector.class_names[int(label_id)]
        conf = float(conf)
        if conf < get_conf_threshold(label_name):
            continue

        box_xyxy = normalize_xyxy(tlbr_to_xyxy(box_tlbr), w, h)
        if box_area_xyxy(box_xyxy) <= 1:
            continue

        detections.append({
            "label": label_name,
            "label_lower": label_name.lower(),
            "confidence": round(conf, 4),
            "box_xyxy": [int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])],
        })

    return remove_nested_duplicates(detections)


# def subtract_masked_trains(
#     full_trains: List[Dict[str, Any]],
#     masked_trains: List[Dict[str, Any]],
#     overlap_thr: float,
# ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     if not masked_trains:
#         return list(full_trains), []
#
#     kept: List[Dict[str, Any]] = []
#     removed: List[Dict[str, Any]] = []
#
#     for det in full_trains:
#         matched = False
#         for ref in masked_trains:
#             iou = iou_xyxy(det["box_xyxy"], ref["box_xyxy"])
#             contain = max(
#                 overlap_ratio_on_small(det["box_xyxy"], ref["box_xyxy"]),
#                 overlap_ratio_on_small(ref["box_xyxy"], det["box_xyxy"]),
#             )
#             if max(iou, contain) >= overlap_thr:
#                 matched = True
#                 break
#
#         if matched:
#             removed.append(det)
#         else:
#             kept.append(det)
#
#     return kept, removed

def subtract_masked_trains(
    full_trains: List[Dict[str, object]],
    masked_trains: List[Dict[str, object]],
    overlap_thr: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not masked_trains:
        return list(full_trains), []

    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def point_in_box(pt, box) -> bool:
        px, py = pt
        x1, y1, x2, y2 = box
        return x1 <= px <= x2 and y1 <= py <= y2

    def center_distance(box1, box2) -> float:
        c1 = box_center(box1)
        c2 = box_center(box2)
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) ** 0.5

    kept: List[Dict[str, object]] = []
    removed: List[Dict[str, object]] = []

    for det in full_trains:
        det_box = det["box_xyxy"]
        matched = False

        for ref in masked_trains:
            ref_box = ref["box_xyxy"]

            iou = iou_xyxy(det_box, ref_box)

            det_in_ref = overlap_ratio_on_small(det_box, ref_box)
            ref_in_det = overlap_ratio_on_small(ref_box, det_box)
            contain = max(det_in_ref, ref_in_det)

            det_center = box_center(det_box)
            ref_center = box_center(ref_box)

            center_inside = (
                point_in_box(det_center, ref_box) or
                point_in_box(ref_center, det_box)
            )

            dist = center_distance(det_box, ref_box)

            # 根据框大小自适应一个中心距离阈值
            x1, y1, x2, y2 = det_box
            det_w = max(1.0, x2 - x1)
            det_h = max(1.0, y2 - y1)
            center_dist_thr = max(20.0, min(det_w, det_h) * 0.5)

            # 满足以下任一情况，就认为 det 和 ref 是同一个目标
            if (
                iou >= overlap_thr
                or (contain >= 0.80 and center_inside)
                or (contain >= overlap_thr and dist <= center_dist_thr)
            ):
                matched = True
                break

        if matched:
            removed.append(det)
        else:
            kept.append(det)

    return kept, removed


# ============================================================
# person 与铁轨关系判定
# ============================================================
def judge_person_rail_v3(
    box_xyxy: Sequence[float],
    polygon: np.ndarray,
    near_threshold: float = 30,
    foot_inset_ratio: float = 0.15,
) -> Dict[str, Any]:
    # 与你给的 person_rail_distance_dual_pass.py 逻辑保持一致
    x1, y1, x2, y2 = box_xyxy
    w = 0

    p_left = (x1 + foot_inset_ratio * w, y2)
    p_right = (x2 - foot_inset_ratio * w, y2)

    d1 = cv2.pointPolygonTest(polygon, p_left, True)
    d2 = cv2.pointPolygonTest(polygon, p_right, True)
    p_mid = ((p_left[0] + p_right[0]) / 2.0, (p_left[1] + p_right[1]) / 2.0)
    d_mid = cv2.pointPolygonTest(polygon, p_mid, True)

    min_distance = min(abs(d1), abs(d2))

    if d1 < 0 and d2 < 0:
        if d_mid >= 0:
            status = "in_rail"
        else:
            status = "near_rail" if min_distance <= near_threshold else "outside"
        distance = min_distance
    elif (d1 >= 0 and d2 < 0) or (d2 >= 0 and d1 < 0):
        status = "on_rail"
        distance = 0.0
    else:
        status = "in_rail"
        distance = 0.0

    return {
        "status": status,
        "distance": float(distance),
        "left_point": p_left,
        "right_point": p_right,
        "mid_point": p_mid,
        "left_dist": float(d1),
        "right_dist": float(d2),
        "mid_dist": float(d_mid),
    }


def analyze_persons(person_dets: List[Dict[str, Any]], rail_polygon: np.ndarray) -> Dict[str, Any]:
    people: List[Dict[str, Any]] = []
    severity_order = {"outside": 0, "near_rail": 1, "on_rail": 2, "in_rail": 3}
    max_status = "outside"
    max_level = 0

    for det in person_dets:
        result = judge_person_rail_v3(
            box_xyxy=det["box_xyxy"],
            polygon=rail_polygon,
            near_threshold=float(CONFIG["person_near_threshold"]),
            foot_inset_ratio=float(CONFIG["person_foot_inset_ratio"]),
        )
        item = dict(det)
        item.update(result)
        people.append(item)

        level = severity_order[result["status"]]
        if level > max_level:
            max_level = level
            max_status = result["status"]

    if max_status in ("on_rail", "in_rail"):
        state_code = STATE_PERSON
    elif max_status == "near_rail":
        state_code = STATE_WARNING
    else:
        state_code = STATE_CLEAR

    return {
        "has_person": bool(people),
        "people": people,
        "max_status": max_status,
        "near_or_in_alarm": max_level >= 1,
        "enter_alarm": max_level >= 2,
        "state_code": state_code,
    }


# ============================================================
# train 与目标铁轨交集判定
# ============================================================
def calc_overlap_with_polygon(
    image_shape: Sequence[int],
    bbox_xyxy: Sequence[int],
    polygon: np.ndarray,
) -> Tuple[int, float, float, np.ndarray]:
    h, w = image_shape[:2]

    box_mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon.reshape((-1, 1, 2))], 255)

    inter_mask = cv2.bitwise_and(box_mask, poly_mask)

    overlap_area = int(np.count_nonzero(inter_mask))
    box_area = max(0, (x2 - x1) * (y2 - y1))
    poly_area = int(np.count_nonzero(poly_mask))

    overlap_ratio_box = overlap_area / (box_area + 1e-6)
    overlap_ratio_poly = overlap_area / (poly_area + 1e-6)

    return overlap_area, float(overlap_ratio_box), float(overlap_ratio_poly), inter_mask


def evaluate_train_boxes(
    frame_shape: Sequence[int],
    train_dets: List[Dict[str, Any]],
    rail_polygon: np.ndarray,
) -> Dict[str, Any]:
    warning_area = int(CONFIG["train_overlap_warning_area"])
    train_area = int(CONFIG["train_overlap_train_area"])

    enriched: List[Dict[str, Any]] = []
    best_det: Optional[Dict[str, Any]] = None
    best_overlap_area = -1
    state_code = STATE_CLEAR

    for det in train_dets:
        overlap_area, overlap_ratio_box, overlap_ratio_poly, _ = calc_overlap_with_polygon(
            frame_shape, det["box_xyxy"], rail_polygon
        )
        det2 = dict(det)
        det2["overlap_area"] = int(overlap_area)
        det2["overlap_ratio_box"] = float(overlap_ratio_box)
        det2["overlap_ratio_poly"] = float(overlap_ratio_poly)

        if overlap_area >= train_area:
            det2["overlap_stage"] = "passing"
            det2["state_code"] = STATE_TRAIN
            state_code = max(state_code, STATE_TRAIN)
        elif overlap_area >= warning_area:
            det2["overlap_stage"] = "entering"
            det2["state_code"] = STATE_WARNING
            state_code = max(state_code, STATE_WARNING)
        else:
            det2["overlap_stage"] = "none"
            det2["state_code"] = STATE_CLEAR

        if overlap_area > best_overlap_area:
            best_overlap_area = overlap_area
            best_det = det2

        enriched.append(det2)

    return {
        "state_code": state_code,
        "best_train": best_det,
        "trains": enriched,
        "max_overlap_area": max(0, best_overlap_area),
    }


# ============================================================
# Canny 轨道占用判定
# ============================================================
def canny_edge(
    img: np.ndarray,
    low: int = 50,
    high: int = 150,
    gaussian_ksize: Tuple[int, int] = (5, 5),
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blur = cv2.GaussianBlur(gray, gaussian_ksize, gaussian_sigma)
    edge = cv2.Canny(blur, low, high)
    return edge


def edge_keep_ratio(edge_ref: np.ndarray, edge_cur: np.ndarray) -> float:
    ref = edge_ref > 0
    cur = edge_cur > 0
    overlap = np.logical_and(ref, cur).sum()
    total = ref.sum() + 1e-6
    return float(overlap / total)


def build_polygon_mask(shape: Sequence[int], polygon: np.ndarray) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], 255)
    return mask


class CannyTrackAnalyzer:
    def __init__(self, rail_polygon: np.ndarray) -> None:
        self.rail_polygon = rail_polygon
        self.ref_path = str(CONFIG.get("edge_reference_path", ""))
        self.ref_img: Optional[np.ndarray] = None
        self.edge_ref: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray] = None
        self.ref_shape: Optional[Tuple[int, int]] = None
        self.ready = False
        self.reload()

    def reload(self) -> bool:
        if not self.ref_path:
            self.ready = False
            log_warning("[canny_reload] skipped: edge_reference_path 为空")
            return False
        ref_img = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None:
            self.ready = False
            log_warning(f"[canny_reload] failed: 无法读取模板 {self.ref_path}")
            return False
        self.ref_img = ref_img
        self.roi_mask = build_polygon_mask(ref_img.shape, self.rail_polygon)
        self.edge_ref = (ref_img > 0).astype(np.uint8) * 255
        self.edge_ref = cv2.bitwise_and(self.edge_ref, self.edge_ref, mask=self.roi_mask)
        self.ref_shape = ref_img.shape[:2]
        self.ready = True
        log_info(f"[canny_reload] success: path={self.ref_path}, shape={self.ref_shape}")
        return True

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.ready or self.edge_ref is None or self.roi_mask is None or self.ref_shape is None:
            return {
                "ready": False,
                "state_code": STATE_CLEAR,
                "status": "not_ready",
                "message": "edge reference not ready",
                "r_keep": 1.0,
                "edge_cur": None,
            }

        ref_h, ref_w = self.ref_shape
        cur_img = frame.copy()
        if cur_img.shape[:2] != (ref_h, ref_w):
            cur_img = cv2.resize(cur_img, (ref_w, ref_h))

        edge_cur = canny_edge(
            cur_img,
            low=int(CONFIG.get("canny_low", 50)),
            high=int(CONFIG.get("canny_high", 150)),
            gaussian_ksize=tuple(CONFIG.get("gaussian_ksize", (5, 5))),
            gaussian_sigma=float(CONFIG.get("gaussian_sigma", 1.0)),
        )
        edge_cur = cv2.bitwise_and(edge_cur, edge_cur, mask=self.roi_mask)
        r_keep = edge_keep_ratio(self.edge_ref, edge_cur)

        low_th = float(CONFIG.get("track_keep_low_th", 0.77))
        high_th = float(CONFIG.get("track_keep_high_th", 0.87))

        if r_keep < low_th:
            state_code = STATE_TRAIN
            status = "occupied"
            message = "Track structure lost: possible train occlusion"
        elif r_keep < high_th:
            state_code = STATE_WARNING
            status = "changed"
            message = "Some changes detected"
        else:
            state_code = STATE_CLEAR
            status = "normal"
            message = "Track structure mostly preserved"

        return {
            "ready": True,
            "state_code": state_code,
            "status": status,
            "message": message,
            "r_keep": float(r_keep),
            "edge_cur": edge_cur,
        }


# ============================================================
# 模板更新
# ============================================================
def build_stable_canny_template_from_frames(
    frames: Sequence[np.ndarray],
    rail_polygon: np.ndarray,
    resize_to: Optional[Tuple[int, int]] = None,
    canny_low: int = 50,
    canny_high: int = 150,
    gaussian_ksize: Tuple[int, int] = (5, 5),
    gaussian_sigma: float = 1.0,
    keep_ratio: float = 0.70,
) -> np.ndarray:
    if not frames:
        raise ValueError("frames 为空，无法构建模板")

    first = frames[0]
    if resize_to is not None:
        first = cv2.resize(first, resize_to)
    roi_mask = build_polygon_mask(first.shape, rail_polygon)

    edge_sum: Optional[np.ndarray] = None
    used = 0

    for frame in frames:
        img = frame
        if resize_to is not None:
            img = cv2.resize(img, resize_to)
        edge = canny_edge(
            img,
            low=canny_low,
            high=canny_high,
            gaussian_ksize=gaussian_ksize,
            gaussian_sigma=gaussian_sigma,
        )
        edge = cv2.bitwise_and(edge, edge, mask=roi_mask)
        edge_binary = (edge > 0).astype(np.uint8)
        if edge_sum is None:
            edge_sum = np.zeros_like(edge_binary, dtype=np.float32)
        edge_sum += edge_binary
        used += 1

    if edge_sum is None or used <= 0:
        raise ValueError("没有成功累积到边缘模板")

    edge_freq = edge_sum / used
    stable_edges = (edge_freq >= keep_ratio).astype(np.uint8) * 255
    return stable_edges


def template_to_binary_float(template: np.ndarray) -> np.ndarray:
    return (template > 0).astype(np.float32)


def compute_template_diff_and_edge_ratio(old_template: np.ndarray, new_template: np.ndarray) -> Tuple[float, float]:
    old_bin = template_to_binary_float(old_template)
    new_bin = template_to_binary_float(new_template)
    if old_bin.shape != new_bin.shape:
        raise ValueError("模板尺寸不一致")
    diff = float(np.mean(np.abs(new_bin - old_bin)))
    edge_ratio = float(np.sum(new_bin) / (np.sum(old_bin) + 1e-6))
    return diff, edge_ratio


def update_canny_template_safe(
    old_template: np.ndarray,
    new_template: np.ndarray,
    prev_fused: Optional[np.ndarray] = None,
    old_weight: float = 0.7,
    new_weight: float = 0.3,
    diff_threshold: float = 0.15,
    binarize_threshold: float = 0.5,
    edge_ratio_low: float = 0.7,
    edge_ratio_high: float = 1.3,
) -> Tuple[np.ndarray, float, float, bool, np.ndarray]:
    old_bin = template_to_binary_float(old_template)
    new_bin = template_to_binary_float(new_template)

    if old_bin.shape != new_bin.shape:
        raise ValueError("模板尺寸不一致")

    diff = float(np.mean(np.abs(new_bin - old_bin)))
    old_edge_count = float(np.sum(old_bin))
    new_edge_count = float(np.sum(new_bin))
    edge_ratio = float(new_edge_count / (old_edge_count + 1e-6))

    allow_update = (
        diff < diff_threshold and
        edge_ratio_low <= edge_ratio <= edge_ratio_high
    )

    if prev_fused is None or prev_fused.shape != new_bin.shape:
        fused_base = old_bin.copy()
    else:
        fused_base = prev_fused.astype(np.float32).copy()

    if allow_update:
        fused = old_weight * fused_base + new_weight * new_bin
        updated = (fused >= binarize_threshold).astype(np.uint8) * 255
        updated_flag = True
    else:
        fused = fused_base
        updated = old_template.copy()
        updated_flag = False

    return updated, diff, edge_ratio, updated_flag, fused


def are_templates_mutually_similar(
    templates: Sequence[np.ndarray],
    diff_threshold: float,
    edge_ratio_low: float,
    edge_ratio_high: float,
) -> Tuple[bool, List[Dict[str, float]]]:
    if len(templates) < 2:
        return False, []

    metrics: List[Dict[str, float]] = []
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            diff_ij, ratio_ij = compute_template_diff_and_edge_ratio(templates[i], templates[j])
            _, ratio_ji = compute_template_diff_and_edge_ratio(templates[j], templates[i])
            metrics.append({
                "pair": f"{i}-{j}",
                "diff": float(diff_ij),
                "edge_ratio_ij": float(ratio_ij),
                "edge_ratio_ji": float(ratio_ji),
            })
            if not (
                diff_ij < diff_threshold and
                edge_ratio_low <= ratio_ij <= edge_ratio_high and
                edge_ratio_low <= ratio_ji <= edge_ratio_high
            ):
                return False, metrics

    return True, metrics


class TemplateUpdater:
    def __init__(self, analyzer: CannyTrackAnalyzer, rail_polygon: np.ndarray) -> None:
        self.analyzer = analyzer
        self.rail_polygon = rail_polygon
        self.enabled = bool(CONFIG.get("template_update_enabled", True))
        self.bootstrap_enabled = bool(CONFIG.get("template_bootstrap_on_start", True))
        self.interval_seconds = float(CONFIG.get("template_update_interval_seconds", 3600))
        self.take_every_n_frames = int(CONFIG.get("template_update_take_every_n_frames", 5))
        self.need_frames = int(CONFIG.get("template_update_need_frames", 20))
        self.keep_ratio = float(CONFIG.get("template_keep_ratio", 0.70))
        self.output_dir = str(CONFIG.get("template_output_dir", "template_update_output"))
        self.template_path = str(CONFIG.get("edge_reference_path", ""))

        self.old_weight = float(CONFIG.get("template_update_old_weight", 0.7))
        self.new_weight = float(CONFIG.get("template_update_new_weight", 0.3))
        self.diff_threshold = float(CONFIG.get("template_update_diff_threshold", 0.15))
        self.binarize_threshold = float(CONFIG.get("template_update_binarize_threshold", 0.5))
        self.edge_ratio_low = float(CONFIG.get("template_update_edge_ratio_low", 0.7))
        self.edge_ratio_high = float(CONFIG.get("template_update_edge_ratio_high", 1.3))

        self.library_max_size = int(CONFIG.get("template_library_max_size", 3))
        self.library_similarity_diff_threshold = float(CONFIG.get("template_library_similarity_diff_threshold", 0.10))
        self.library_similarity_edge_ratio_low = float(CONFIG.get("template_library_similarity_edge_ratio_low", 0.85))
        self.library_similarity_edge_ratio_high = float(CONFIG.get("template_library_similarity_edge_ratio_high", 1.15))

        self.collecting = False
        self.collected_frames: List[np.ndarray] = []
        self.collect_frame_counter = 0
        self.last_update_ts = time.time()
        self.bootstrap_completed = bool(self.analyzer.ready)
        self.bootstrap_skip_logged = False

        self.fused_path = os.path.join(self.output_dir, "template_fused.npy")
        self.candidate_library_dir = os.path.join(self.output_dir, "template_update_library")
        self.candidate_index_path = os.path.join(self.candidate_library_dir, "library_index.json")
        self.fused_state: Optional[np.ndarray] = None
        self.failed_update_count = 0
        self.candidate_seq = 0
        self.candidate_records: List[Dict[str, Any]] = []
        ################################################################
        self.bad_template_bypass_update_interval_on_trigger = bool(
            CONFIG.get("bad_template_bypass_update_interval_on_trigger", True)
        )

        ensure_dir(self.output_dir)
        ensure_dir(self.candidate_library_dir)
        if self.template_path:
            ensure_dir(os.path.dirname(self.template_path))

        self._sync_fused_with_current_template()

    def reset_collection(self) -> None:
        self.collecting = False
        self.collected_frames = []
        self.collect_frame_counter = 0

    def _current_template_binary(self) -> Optional[np.ndarray]:
        if self.analyzer.ref_img is None:
            return None
        return template_to_binary_float(self.analyzer.ref_img)

    def _save_fused_state(self, fused: np.ndarray) -> None:
        ensure_dir(os.path.dirname(self.fused_path))
        np.save(self.fused_path, fused.astype(np.float32))

    def _sync_fused_with_current_template(self) -> None:
        current_bin = self._current_template_binary()
        if current_bin is None:
            self.fused_state = None
            return

        fused: Optional[np.ndarray] = None
        if os.path.isfile(self.fused_path):
            try:
                loaded = np.load(self.fused_path)
                if loaded.shape == current_bin.shape:
                    fused = loaded.astype(np.float32)
                else:
                    log_warning(
                        f"[template_fused] shape mismatch, reset fused: loaded={loaded.shape}, current={current_bin.shape}"
                    )
            except Exception as exc:
                log_warning(f"[template_fused] load failed, reset fused: {exc}")

        if fused is None:
            fused = current_bin.copy()
            try:
                self._save_fused_state(fused)
            except Exception as exc:
                log_warning(f"[template_fused] init save failed: {exc}")

        self.fused_state = fused

    def _remove_file_if_exists(self, path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    def _write_candidate_index(self) -> None:
        records = []
        for item in self.candidate_records:
            records.append({
                "created_at": item["created_at"],
                "path": item["path"],
                "diff_from_base": float(item["diff_from_base"]),
                "edge_ratio_from_base": float(item["edge_ratio_from_base"]),
            })
        with open(self.candidate_index_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _clear_candidate_library(self, keep_dir: bool = True) -> None:
        for item in self.candidate_records:
            self._remove_file_if_exists(item.get("path", ""))
        self.candidate_records = []
        self.failed_update_count = 0
        self._remove_file_if_exists(self.candidate_index_path)
        if not keep_dir:
            try:
                if os.path.isdir(self.candidate_library_dir):
                    os.rmdir(self.candidate_library_dir)
            except Exception:
                pass

    def _push_candidate_to_library(self, candidate_template: np.ndarray, diff: float, edge_ratio: float) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.candidate_seq += 1
        candidate_path = os.path.join(
            self.candidate_library_dir,
            f"candidate_{timestamp}_{self.candidate_seq:03d}.png",
        )
        ok = cv2.imwrite(candidate_path, candidate_template)
        if not ok:
            raise RuntimeError(f"写入候选模板失败: {candidate_path}")

        self.candidate_records.append({
            "created_at": timestamp,
            "path": candidate_path,
            "diff_from_base": float(diff),
            "edge_ratio_from_base": float(edge_ratio),
            "template": candidate_template.copy(),
        })

        while len(self.candidate_records) > self.library_max_size:
            removed = self.candidate_records.pop(0)
            self._remove_file_if_exists(removed.get("path", ""))

        self._write_candidate_index()

    def _save_template(
        self,
        new_template: np.ndarray,
        fused_state: np.ndarray,
        now_ts: float,
        mode: str,
        resize_to: Optional[Tuple[int, int]],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.output_dir, f"stable_edge_template_{timestamp}.png")
        ok_backup = cv2.imwrite(backup_path, new_template)
        ok_main = False
        if self.template_path:
            ok_main = cv2.imwrite(self.template_path, new_template)

        if self.template_path and not ok_main:
            raise RuntimeError(f"写入模板失败: {self.template_path}")
        if not ok_backup:
            raise RuntimeError(f"写入模板备份失败: {backup_path}")

        self._save_fused_state(fused_state)
        self.fused_state = fused_state.astype(np.float32).copy()

        reloaded = self.analyzer.reload()
        if self.template_path and not reloaded:
            raise RuntimeError(f"模板写入后重新加载失败: {self.template_path}")

        self.last_update_ts = now_ts
        self.bootstrap_completed = True
        self.reset_collection()
        self._clear_candidate_library()

        meta: Dict[str, Any] = {
            "mode": mode,
            "updated_at": timestamp,
            "backup_path": backup_path,
            "template_path": self.template_path,
            "fused_path": self.fused_path,
            "need_frames": self.need_frames,
            "take_every_n_frames": self.take_every_n_frames,
            "resize_to": list(resize_to) if resize_to is not None else None,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
            "diff_threshold": self.diff_threshold,
            "binarize_threshold": self.binarize_threshold,
            "edge_ratio_low": self.edge_ratio_low,
            "edge_ratio_high": self.edge_ratio_high,
        }
        if extra_info:
            meta.update(extra_info)

        with open(os.path.join(self.output_dir, f"template_update_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if mode == "bootstrap":
            return f"[template_bootstrap] initial template created with {self.need_frames} clean frames"
        if mode == "update":
            return f"[template] updated with {self.need_frames} clean frames"
        if mode == "forced_library_replace":
            return "[template] original template abandoned: replaced with latest similar template from update library"
        if mode == "forced_rebuild":
            return "[template] original template abandoned: rebuilt template because update library candidates were not similar"
        if mode == "bad_template_rebuild":
            return "[template] bad template recovery triggered: rebuilt template from clean frames"
        return f"[template] template saved, mode={mode}"

    def _try_safe_update(
        self,
        new_template: np.ndarray,
        now_ts: float,
        resize_to: Optional[Tuple[int, int]],
        mode: str,
    ) -> str:
        old_template = self.analyzer.ref_img
        if old_template is None:
            fused = template_to_binary_float(new_template)
            return self._save_template(
                new_template=new_template,
                fused_state=fused,
                now_ts=now_ts,
                mode=mode,
                resize_to=resize_to,
                extra_info={"reason": "no_old_template"},
            )

        updated, diff, edge_ratio, updated_flag, fused = update_canny_template_safe(
            old_template=old_template,
            new_template=new_template,
            prev_fused=self.fused_state,
            old_weight=self.old_weight,
            new_weight=self.new_weight,
            diff_threshold=self.diff_threshold,
            binarize_threshold=self.binarize_threshold,
            edge_ratio_low=self.edge_ratio_low,
            edge_ratio_high=self.edge_ratio_high,
        )

        if updated_flag:
            return self._save_template(
                new_template=updated,
                fused_state=fused,
                now_ts=now_ts,
                mode=mode,
                resize_to=resize_to,
                extra_info={
                    "diff": float(diff),
                    "edge_ratio": float(edge_ratio),
                    "updated_flag": True,
                    "reason": "safe_update_passed",
                },
            )

        self.failed_update_count += 1
        self._push_candidate_to_library(new_template, diff, edge_ratio)

        log_warning(
            f"[template] safe update rejected: diff={diff:.6f}, edge_ratio={edge_ratio:.6f}, "
            f"failed_count={self.failed_update_count}, library_size={len(self.candidate_records)}"
        )

        if self.failed_update_count >= self.library_max_size and len(self.candidate_records) >= self.library_max_size:
            recent_records = self.candidate_records[-self.library_max_size:]
            candidate_templates = [item["template"] for item in recent_records]
            similar, pair_metrics = are_templates_mutually_similar(
                candidate_templates,
                diff_threshold=self.library_similarity_diff_threshold,
                edge_ratio_low=self.library_similarity_edge_ratio_low,
                edge_ratio_high=self.library_similarity_edge_ratio_high,
            )

            if similar:
                latest_template = recent_records[-1]["template"]
                forced_fused = template_to_binary_float(latest_template)
                forced_updated = (forced_fused >= self.binarize_threshold).astype(np.uint8) * 255
                return self._save_template(
                    new_template=forced_updated,
                    fused_state=forced_fused,
                    now_ts=now_ts,
                    mode="forced_library_replace",
                    resize_to=resize_to,
                    extra_info={
                        "diff": float(diff),
                        "edge_ratio": float(edge_ratio),
                        "updated_flag": False,
                        "reason": "library_similar_after_three_failed_updates",
                        "library_pair_metrics": pair_metrics,
                    },
                )

            rebuilt_template = new_template.copy()
            rebuilt_fused = template_to_binary_float(rebuilt_template)
            rebuilt_updated = (rebuilt_fused >= self.binarize_threshold).astype(np.uint8) * 255
            return self._save_template(
                new_template=rebuilt_updated,
                fused_state=rebuilt_fused,
                now_ts=now_ts,
                mode="forced_rebuild",
                resize_to=resize_to,
                extra_info={
                    "diff": float(diff),
                    "edge_ratio": float(edge_ratio),
                    "updated_flag": False,
                    "reason": "library_not_similar_after_three_failed_updates",
                    "library_pair_metrics": pair_metrics,
                },
            )

        return (
            f"[template] candidate stored: diff={diff:.6f}, edge_ratio={edge_ratio:.6f}, "
            f"failed_count={self.failed_update_count}, library_size={len(self.candidate_records)}"
        )

    def _collect_and_build(
        self,
        frame: np.ndarray,
        target_clear: bool,
        clear_reason: str,
        mode: str,
        resize_to: Optional[Tuple[int, int]],
        force_rebuild: bool = False,
    ) -> Optional[str]:
        if not target_clear:
            if self.collecting:
                self.reset_collection()
                if mode == "bootstrap":
                    return f"[template_bootstrap] abort: target rail not clear ({clear_reason})"
                return f"[template] abort update: target rail not clear ({clear_reason})"
            return None

        if not self.collecting:
            self.collecting = True
            self.collected_frames = []
            self.collect_frame_counter = 0
            if mode == "bootstrap":
                return "[template_bootstrap] start collecting clean frames"
            return "[template] start collecting clean frames"

        self.collect_frame_counter += 1
        if (self.collect_frame_counter - 1) % self.take_every_n_frames == 0:
            self.collected_frames.append(frame.copy())
            if len(self.collected_frames) >= self.need_frames:
                new_template = build_stable_canny_template_from_frames(
                    frames=self.collected_frames,
                    rail_polygon=self.rail_polygon,
                    resize_to=resize_to,
                    canny_low=int(CONFIG.get("canny_low", 50)),
                    canny_high=int(CONFIG.get("canny_high", 150)),
                    gaussian_ksize=tuple(CONFIG.get("gaussian_ksize", (5, 5))),
                    gaussian_sigma=float(CONFIG.get("gaussian_sigma", 1.0)),
                    keep_ratio=self.keep_ratio,
                )

                now_ts = time.time()
                if mode == "bootstrap":
                    fused = template_to_binary_float(new_template)
                    return self._save_template(
                        new_template=new_template,
                        fused_state=fused,
                        now_ts=now_ts,
                        mode="bootstrap",
                        resize_to=resize_to,
                        extra_info={"reason": "bootstrap_initial_template"},
                    )

                if force_rebuild:
                    rebuilt_fused = template_to_binary_float(new_template)
                    rebuilt_updated = (rebuilt_fused >= self.binarize_threshold).astype(np.uint8) * 255
                    return self._save_template(
                        new_template=rebuilt_updated,
                        fused_state=rebuilt_fused,
                        now_ts=now_ts,
                        mode="bad_template_rebuild",
                        resize_to=resize_to,
                        extra_info={"reason": "bad_template_recovery_forced_rebuild"},
                    )

                return self._try_safe_update(
                    new_template=new_template,
                    now_ts=now_ts,
                    resize_to=resize_to,
                    mode="update",
                )

        return None

    def maybe_update(
        self,
        frame: np.ndarray,
        target_clear: bool,
        clear_reason: str,
        force_rebuild: bool = False,
    ) -> Optional[str]:
        if not self.analyzer.ready:
            if not self.bootstrap_enabled:
                if not self.bootstrap_skip_logged:
                    self.bootstrap_skip_logged = True
                    return "[template_bootstrap] skipped: analyzer not ready and template_bootstrap_on_start=False"
                return None
            resize_to = (frame.shape[1], frame.shape[0])
            return self._collect_and_build(
                frame=frame,
                target_clear=target_clear,
                clear_reason=clear_reason,
                mode="bootstrap",
                resize_to=resize_to,
                force_rebuild=False,
            )

        if not self.enabled:
            return None

        # now_ts = time.time()
        # if not self.collecting and (now_ts - self.last_update_ts) < self.interval_seconds:
        #     return None

        now_ts = time.time()

        bypass_interval = (
                force_rebuild and
                self.bad_template_bypass_update_interval_on_trigger
        )

        if not self.collecting and (not bypass_interval) and (now_ts - self.last_update_ts) < self.interval_seconds:
            return None

        resize_to = (self.analyzer.ref_shape[1], self.analyzer.ref_shape[0]) if self.analyzer.ref_shape else None
        return self._collect_and_build(
            frame=frame,
            target_clear=target_clear,
            clear_reason=clear_reason,
            mode="update",
            resize_to=resize_to,
            force_rebuild=force_rebuild,
        )


# ============================================================
# 稳定器 / 上报
# ============================================================
class StableStateTracker:
    def __init__(self, required_samples: int) -> None:
        self.required_samples = max(1, int(required_samples))
        self.history: Deque[int] = deque(maxlen=self.required_samples)
        self.stable_state = STATE_CLEAR

    def update(self, raw_state: int) -> Tuple[int, bool]:
        self.history.append(int(raw_state))
        changed = False
        if len(self.history) == self.required_samples and len(set(self.history)) == 1:
            new_state = self.history[-1]
            if new_state != self.stable_state:
                self.stable_state = new_state
                changed = True
        return self.stable_state, changed


class StatusPusher:
    def __init__(self) -> None:
        self.enabled = bool(CONFIG.get("api_push_enabled", False))
        self.heartbeat_enabled = bool(CONFIG.get("heartbeat_enabled", True))
        self.heartbeat_init_status = int(CONFIG.get("heartbeat_init_status", -1))
        self.heartbeat_interval_seconds = float(CONFIG.get("heartbeat_interval_seconds", 300))
        self.last_pushed_status: Optional[int] = None
        self.last_heartbeat_attempt_ts: float = 0.0
        self.startup_init_pushed = False

    def _build_request(self, status_code: int) -> Tuple[str, Dict[str, Any]]:
        api_cfg = CONFIG.get("external_api", {})
        base_url = str(api_cfg.get("base_url", "")).rstrip("/")
        endpoint = str(api_cfg.get("endpoint", ""))
        url = f"{base_url}{endpoint}"
        input_mode = str(CONFIG.get("input_mode", "video"))
        if input_mode == "camera":
            device_id = CONFIG.get("camera", {}).get("device_id", 0)
        else:
            device_id = 0

        params = {
            str(api_cfg.get("device_id_param", "deviceId")): device_id,
            str(api_cfg.get("status_param", "railStatus")): int(status_code),
        }
        extra_params = api_cfg.get("extra_params", {})
        if isinstance(extra_params, dict):
            params.update(extra_params)
        return url, params

    def push_status(self, status_code: int, reason: str = "manual", force: bool = False) -> bool:
        if not self.enabled:
            return False
        if (not force) and self.last_pushed_status == int(status_code):
            return False

        api_cfg = CONFIG.get("external_api", {})
        url, params = self._build_request(int(status_code))

        log_info(f"[push_request] reason={reason}, status={int(status_code)}, url={url}, params={params}")
        try:
            response = requests.get(url, params=params, timeout=float(api_cfg.get("timeout", 5)))
            body = (response.text or "").strip()
            ok = response.status_code == 200
            log_info(
                f"[push_response] reason={reason}, status={int(status_code)}, url={response.url}, "
                # f"code={response.status_code}, body={body}"
                f"code={response.status_code}"
            )
            if ok:
                self.last_pushed_status = int(status_code)
            log_info(f"[push_result] reason={reason}, success={ok}, pushed_status={int(status_code)}")
            return ok
        except Exception as exc:
            log_error(f"[push_error] reason={reason}, status={int(status_code)}, error={exc}")
            return False

    def push_if_changed(self, status_code: int) -> bool:
        return self.push_status(int(status_code), reason="status_change", force=False)

    def push_startup_init(self) -> bool:
        if self.startup_init_pushed:
            return False
        ok = self.push_status(self.heartbeat_init_status, reason="startup_init", force=True)
        self.startup_init_pushed = True
        self.last_heartbeat_attempt_ts = time.time()
        return ok

    def maybe_push_heartbeat(self, current_status: int) -> bool:
        if not self.enabled or not self.heartbeat_enabled:
            return False

        now_ts = time.time()
        if self.last_heartbeat_attempt_ts > 0 and (now_ts - self.last_heartbeat_attempt_ts) < self.heartbeat_interval_seconds:
            return False

        self.last_heartbeat_attempt_ts = now_ts
        return self.push_status(int(current_status), reason="heartbeat", force=True)

# ============================================================
# 摄像头 RTSP 输入
# ============================================================

class RTSPStreamReader:
    def __init__(self, rtsp_url: str, camera_name: str = "camera") -> None:
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.thread_name = f"[{camera_name}]"

        self.grab_interval = float(CONFIG.get("grab_interval", 0.05))
        self.frame_stale_seconds = float(CONFIG.get("frame_stale_seconds", 2.0))
        self.cooldown_network = float(CONFIG.get("cooldown_network_seconds", 60))
        self.cooldown_service = float(CONFIG.get("cooldown_service_seconds", 30))

        # 按 connection.py 的思路：小缓冲 + 明确超时 + 持续 grab
        self.buffer_size = 1
        self.open_timeout_msec = 3000
        self.read_timeout_msec = 1000

        # 用于限制 retrieve 频率，避免无意义过快取帧
        self.target_fps = max(1.0, 1.0 / max(0.001, self.grab_interval))
        self.min_publish_interval = 1.0 / self.target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.cap_lock = threading.RLock()

        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_ts: float = 0.0
        self.latest_frame_id: int = 0

        self.new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)

        self.connection_state = CONNECTION_STARTING
        self.error_state: Optional[str] = None
        self.error_timestamp: float = 0.0
        self.last_reconnect_ts: float = 0.0
        self.is_first_run = True
        self.connection_attempts = 0

        self.last_publish_ts: float = 0.0
        self.empty_grab_count: int = 0

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()
            log_info(f"{self.thread_name} 抓流线程已启动")

    def stop(self) -> None:
        self._stop_event.set()
        self.new_frame_event.set()
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
        except Exception:
            pass
        self._release_capture()
        self._clear_latest_frame()

    def _set_latest_frame(self, frame: np.ndarray, ts: float) -> None:
        with self.frame_lock:
            self.latest_frame = frame
            self.latest_frame_ts = ts
            self.latest_frame_id += 1
        self.new_frame_event.set()

    def _clear_latest_frame(self) -> None:
        with self.frame_lock:
            self.latest_frame = None
            self.latest_frame_ts = 0.0
        self.new_frame_event.clear()

    def _release_capture(self) -> None:
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _check_ping(self) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.rtsp_url)
            host = parsed.hostname
            if not host:
                return False

            system = platform.system().lower()
            if "windows" in system:
                cmd = ["ping", "-n", "1", "-w", "1000", host]
            else:
                cmd = ["ping", "-c", "1", "-W", "1", host]

            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _open_capture(self) -> bool:
        with self.cap_lock:
            if self.cap is not None and self.cap.isOpened():
                return True

            self._release_capture()
            self.connection_attempts += 1

            try:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if cap is None or not cap.isOpened():
                    self.cap = None
                    return False

                # 按 connection.py 的方式设置
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_msec)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_msec)

                # 首帧验证：这是 connection.py 的关键点
                ret, frame = cap.read()
                if not ret or frame is None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    return False

                self.cap = cap

                # 首帧直接发布一次，避免主循环刚启动时空等
                now_ts = time.time()
                self._set_latest_frame(frame, now_ts)
                self.last_publish_ts = now_ts
                self.empty_grab_count = 0

                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if self.is_first_run:
                    log_info(
                        f"{self.thread_name} RTSP连接成功: "
                        f"{actual_width}x{actual_height} @ {actual_fps:.1f}fps, "
                        f"buffer={self.buffer_size}"
                    )
                else:
                    log_info(
                        f"{self.thread_name} RTSP重连成功 "
                        f"(attempt={self.connection_attempts}): "
                        f"{actual_width}x{actual_height} @ {actual_fps:.1f}fps"
                    )

                self.connection_state = CONNECTION_ONLINE
                self.error_state = None
                self.last_reconnect_ts = now_ts
                self.connection_attempts = 0
                return True

            except Exception as exc:
                self.cap = None
                log_warning(f"{self.thread_name} 打开RTSP失败: {exc}")
                return False

    def _read_latest_frame_from_capture(self) -> Optional[np.ndarray]:
        """
        按 connection.py 的思想：
        - 连接保持常驻
        - 持续 grab
        - 到了最小发布间隔再 retrieve
        - 只保留最新帧
        """
        with self.cap_lock:
            cap = self.cap
            if cap is None or not cap.isOpened():
                return None

            try:
                grabbed = cap.grab()
                if not grabbed:
                    return None

                now_ts = time.time()
                if (now_ts - self.last_publish_ts) < self.min_publish_interval:
                    return None

                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    return None

                self.last_publish_ts = now_ts
                return frame

            except Exception:
                return None

    def _handle_connection_error(self) -> None:
        self.error_timestamp = time.time()
        self._release_capture()
        self._clear_latest_frame()
        self.connection_state = CONNECTION_OFFLINE

        log_warning(f"{self.thread_name} 抓流失败，开始诊断连接状态")
        if self._check_ping():
            self.error_state = "service"
            log_warning(f"{self.thread_name} Ping通，判定为服务/流故障，冷却 {self.cooldown_service:.0f}s 后重试")
        else:
            self.error_state = "network"
            log_warning(f"{self.thread_name} Ping不通，判定为网络故障，冷却 {self.cooldown_network:.0f}s 后重试")

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.error_state:
                cooldown = self.cooldown_network if self.error_state == "network" else self.cooldown_service
                if time.time() - self.error_timestamp < cooldown:
                    time.sleep(0.2)
                    continue

            if not self._open_capture():
                self._handle_connection_error()
                time.sleep(0.5)
                continue

            frame = self._read_latest_frame_from_capture()

            if frame is not None:
                now_ts = time.time()
                self._set_latest_frame(frame, now_ts)

                if self.is_first_run:
                    log_info(f"{self.thread_name} 初始化连接成功，开始持续抓取图像")
                    self.is_first_run = False

                if self.connection_state != CONNECTION_ONLINE:
                    log_info(f"{self.thread_name} 重连成功，恢复在线")
                    self.connection_state = CONNECTION_ONLINE
                    self.error_state = None
                    self.last_reconnect_ts = now_ts

                self.empty_grab_count = 0
                time.sleep(0.001)
                continue

            # 这里不再像旧版那样马上走 ffmpeg fallback
            # 而是允许短时间 grab/retrieve 空跑，保持长连接
            self.empty_grab_count += 1

            if self.empty_grab_count >= 50:
                log_warning(f"{self.thread_name} 连续读取空帧过多，准备重连")
                self.empty_grab_count = 0
                self._handle_connection_error()
                time.sleep(0.2)
                continue

            time.sleep(0.005)

    def wait_for_newer_frame(self, last_frame_id: int, timeout: float = 0.2) -> bool:
        deadline = time.time() + max(0.0, float(timeout))
        while not self._stop_event.is_set():
            with self.frame_lock:
                if self.latest_frame is not None and self.latest_frame_id > int(last_frame_id):
                    return True

            remain = deadline - time.time()
            if remain <= 0:
                return False

            self.new_frame_event.wait(timeout=remain)
            self.new_frame_event.clear()

        return False

    def get_latest_frame(self, max_age: Optional[float] = None) -> Tuple[Optional[np.ndarray], float, int]:
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            ts = self.latest_frame_ts
            frame_id = self.latest_frame_id

        if frame is None:
            return None, ts, frame_id

        age_limit = self.frame_stale_seconds if max_age is None else float(max_age)
        if age_limit > 0 and ts and (time.time() - ts > age_limit):
            return None, ts, frame_id

        return frame, ts, frame_id

    def get_status(self) -> Dict[str, Any]:
        with self.frame_lock:
            latest_ts = self.latest_frame_ts
            latest_id = self.latest_frame_id

        return {
            "connection_state": self.connection_state,
            "error_state": self.error_state,
            "last_reconnect_ts": self.last_reconnect_ts,
            "latest_frame_ts": latest_ts,
            "latest_frame_id": latest_id,
        }

    def release(self) -> None:
        self.stop()


def confirm_train_state_with_template(
    train_box_summary: Dict[str, Any],
    train_canny_summary: Dict[str, Any],
) -> Tuple[int, bool, str]:
    """
    box 先给出“目标轨道是否有车”的候选结果；
    只有当模板也认为“有车/有明显变化”时，才最终保留 box 的 train 状态。
    """
    box_state = int(train_box_summary.get("state_code", STATE_CLEAR))
    canny_state = int(train_canny_summary.get("state_code", STATE_CLEAR))
    canny_ready = bool(train_canny_summary.get("ready", False))

    if box_state == STATE_CLEAR:
        return STATE_CLEAR, False, "box_clear"

    if not bool(CONFIG.get("train_template_confirm_enabled", True)):
        return box_state, True, "template_confirm_disabled"

    if not canny_ready:
        return box_state, False, "template_not_ready_fallback_box"

    accept_warning = bool(CONFIG.get("train_template_confirmation_accept_warning", True))
    template_has_train = (canny_state != STATE_CLEAR) if accept_warning else (canny_state == STATE_TRAIN)

    if template_has_train:
        return box_state, True, f"confirmed_by_template_{state_name(canny_state)}"

    return STATE_CLEAR, False, f"rejected_by_template_{state_name(canny_state)}"


# ============================================================
# 结果结构
# ============================================================
@dataclass
class FrameDecision:
    persons: List[Dict[str, Any]] = field(default_factory=list)
    trains: List[Dict[str, Any]] = field(default_factory=list)
    removed_trains: List[Dict[str, Any]] = field(default_factory=list)
    person_summary: Dict[str, Any] = field(default_factory=dict)
    train_box_summary: Dict[str, Any] = field(default_factory=dict)
    train_canny_summary: Dict[str, Any] = field(default_factory=dict)
    train_box_state_raw: int = STATE_CLEAR
    train_canny_state_raw: int = STATE_CLEAR
    train_template_confirmed: bool = False
    train_template_reason: str = ""
    train_state_raw: int = STATE_CLEAR
    person_state_raw: int = STATE_CLEAR
    train_state_stable: int = STATE_CLEAR
    person_state_stable: int = STATE_CLEAR
    final_state_raw: int = STATE_CLEAR
    final_state_stable: int = STATE_CLEAR
    sampled: bool = False
    frame_index: int = 0
    status_changed: bool = False
    clear_for_template_update: bool = True
    train_box_in_target: bool = False
    canny_alarm: bool = False
    bad_template_suspect_count: int = 0
    bad_template_recovery_active: bool = False


# ============================================================
# 主监控器
# ============================================================
class RailwayMonitor:
    def __init__(self) -> None:
        self.detector = load_detector()
        self.target_rail_polygon = to_polygon(CONFIG["target_rail_polygon"])
        self.mask_polygons = polygons_from_config(CONFIG.get("mask_polygons", []))
        self.person_labels = list(CONFIG.get("person_labels", ["person"]))
        self.train_labels = list(CONFIG.get("train_labels", ["train"]))
        self.detect_stride = int(CONFIG.get("detect_interval_frames", 0)) + 1
        self.person_tracker = StableStateTracker(int(CONFIG.get("stable_required_samples", 3)))
        self.train_tracker = StableStateTracker(int(CONFIG.get("stable_required_samples", 3)))
        self.status_pusher = StatusPusher()
        self.canny_analyzer = CannyTrackAnalyzer(self.target_rail_polygon)
        self.template_updater = TemplateUpdater(self.canny_analyzer, self.target_rail_polygon)
        self.last_decision = FrameDecision()
        self.last_final_stable_state = STATE_CLEAR
        self.bad_template_recovery_enabled = bool(CONFIG.get("bad_template_recovery_enabled", True))
        self.bad_template_suspect_threshold = max(1, int(CONFIG.get("bad_template_suspect_threshold", 30)))
        self.bad_template_force_rebuild_on_trigger = bool(CONFIG.get("bad_template_force_rebuild_on_trigger", True))
        self.bad_template_suspect_count = 0
        self.bad_template_recovery_active = False

        if self.canny_analyzer.ready:
            log_info(f"[startup] 使用现有 Canny 模板: {self.canny_analyzer.ref_path}")
        else:
            if bool(CONFIG.get("template_bootstrap_on_start", True)):
                log_warning("[startup] 未检测到可用 Canny 模板，将在运行开始后自动采集干净帧生成初始模板")
            else:
                log_warning("[startup] 未检测到可用 Canny 模板，且已关闭启动自动建模，Canny 检测将不可用")
        log_info(
            f"[monitor_init] detect_stride={self.detect_stride}, "
            f"canny_ready={self.canny_analyzer.ready}, template_update_enabled={CONFIG.get('template_update_enabled', True)}"
        )
        self.status_pusher.push_startup_init()

    def process_sampled_frame(
            self,
            frame: np.ndarray,
            frame_index: int,
            source_frame_id: Optional[int] = None,
            frame_ts: Optional[float] = None,
    ) -> FrameDecision:
        masked_frame = apply_black_mask(frame, self.mask_polygons) if self.mask_polygons else frame.copy()
        detections_masked = detect_frame_raw(self.detector, masked_frame)
        detections_full = detect_frame_raw(self.detector, frame)

        masked_trains = filter_by_labels(detections_masked, self.train_labels)
        full_trains = filter_by_labels(detections_full, self.train_labels)
        persons_full = filter_by_labels(detections_full, self.person_labels)

        filtered_train_dets, removed_train_dets = subtract_masked_trains(
            full_trains,
            masked_trains,
            float(CONFIG["same_train_overlap_threshold"]),
        )
        filtered_train_dets = remove_nested_duplicates(filtered_train_dets)

        person_summary = analyze_persons(persons_full, self.target_rail_polygon)
        train_box_summary = evaluate_train_boxes(frame.shape, filtered_train_dets, self.target_rail_polygon)
        train_canny_summary = self.canny_analyzer.analyze(frame)

        person_state_raw = int(person_summary.get("state_code", STATE_CLEAR))
        train_box_state_raw = int(train_box_summary.get("state_code", STATE_CLEAR))
        train_canny_state_raw = int(train_canny_summary.get("state_code", STATE_CLEAR))

        train_state_raw, train_template_confirmed, train_template_reason = confirm_train_state_with_template(
            train_box_summary=train_box_summary,
            train_canny_summary=train_canny_summary,
        )

        #新加用于最终状态绘制框部分
        visual_trains = []
        for det in train_box_summary.get("trains", []):
            det_vis = dict(det)
            raw_box_state = int(det_vis.get("state_code", STATE_CLEAR))

            # 默认先沿用框和目标铁轨区域重叠得到的结果
            det_vis["final_state_code"] = raw_box_state
            det_vis["final_stage"] = det_vis.get("overlap_stage", "none")

            # 只有“框判定命中目标铁轨”的 train，才受模板二次确认影响
            # if raw_box_state != STATE_CLEAR:
            #     if train_state_raw == STATE_CLEAR:
            #         # 模板二次确认否掉
            #         det_vis["final_state_code"] = STATE_CLEAR
            #         det_vis["final_stage"] = "rejected_by_template"
            #     else:
            #         # 模板二次确认通过
            #         det_vis["final_state_code"] = raw_box_state
            #         det_vis["final_stage"] = f"{det_vis.get('overlap_stage', 'none')}_confirmed"

            det_vis["template_reason"] = train_template_reason
            det_vis["canny_state_code"] = train_canny_state_raw
            visual_trains.append(det_vis)

        person_state_stable, _ = self.person_tracker.update(person_state_raw)
        train_state_stable, _ = self.train_tracker.update(train_state_raw)

        final_state_raw = max(person_state_raw, train_state_raw)
        final_state_stable = max(person_state_stable, train_state_stable)

        status_changed = final_state_stable != self.last_final_stable_state
        if status_changed:
            self.last_final_stable_state = final_state_stable
            log_info(
                f"[STATUS_CHANGE] frame={frame_index}, src_id={source_frame_id}, "
                f"raw={final_state_raw}, stable={final_state_stable}, "
                f"person_raw={person_state_raw}, train_raw={train_state_raw}, "
                f"train_box_raw={train_box_state_raw}, train_canny_raw={train_canny_state_raw}, "
                f"train_template_confirmed={int(train_template_confirmed)}, "
                f"person_stable={person_state_stable}, train_stable={train_state_stable}"
            )
            self.status_pusher.push_if_changed(final_state_stable)

        person_in_target = person_summary.get("max_status") in ("on_rail", "in_rail")
        train_box_in_target = int(train_box_summary.get("state_code", STATE_CLEAR)) != STATE_CLEAR
        canny_alarm = int(train_canny_summary.get("state_code", STATE_CLEAR)) != STATE_CLEAR

        clear_for_template_update = (not person_in_target and not train_box_in_target)

        bad_template_suspected = (
                self.bad_template_recovery_enabled and
                (not person_in_target) and
                (not train_box_in_target) and
                canny_alarm
        )

        if bad_template_suspected:
            self.bad_template_suspect_count += 1
            if (
                    self.bad_template_suspect_count in (1, self.bad_template_suspect_threshold)
                    or self.bad_template_suspect_count % 10 == 0
            ):
                log_warning(
                    f"[bad_template_suspect] frame={frame_index}, src_id={source_frame_id}, "
                    f"count={self.bad_template_suspect_count}, "
                    f"box_state={state_name(train_box_summary.get('state_code', STATE_CLEAR))}, "
                    f"canny_state={state_name(train_canny_summary.get('state_code', STATE_CLEAR))}, "
                    f"r_keep={train_canny_summary.get('r_keep', 1.0):.4f}"
                )
        else:
            if self.bad_template_suspect_count > 0:
                log_info(f"[bad_template_suspect] reset count from {self.bad_template_suspect_count} to 0")
            self.bad_template_suspect_count = 0

        prev_bad_template_recovery_active = self.bad_template_recovery_active
        self.bad_template_recovery_active = (
                self.bad_template_recovery_enabled and
                self.bad_template_force_rebuild_on_trigger and
                self.bad_template_suspect_count >= self.bad_template_suspect_threshold
        )

        if self.bad_template_recovery_active and not prev_bad_template_recovery_active:
            log_warning(
                f"[bad_template_recovery] active at frame={frame_index}, count={self.bad_template_suspect_count}, "
                "template rebuild will bypass safe diff check once enough clean frames are collected"
            )
        elif (not self.bad_template_recovery_active) and prev_bad_template_recovery_active:
            log_info("[bad_template_recovery] cleared")

        frame_age_ms = -1.0
        if frame_ts:
            frame_age_ms = max(0.0, (time.time() - float(frame_ts)) * 1000.0)

        log_info(
            f"[sample] frame={frame_index}, src_id={source_frame_id}, "
            f"frame_age_ms={frame_age_ms:.1f}, "
            f"person_raw={state_name(person_state_raw)}, train_raw={state_name(train_state_raw)}, "
            f"train_box_raw={state_name(train_box_state_raw)}, train_canny_raw={state_name(train_canny_state_raw)}, "
            f"train_template_confirmed={int(train_template_confirmed)}, train_template_reason={train_template_reason}, "
            f"person_stable={state_name(person_state_stable)}, train_stable={state_name(train_state_stable)}, "
            f"final_stable={state_name(final_state_stable)}, "
            f"overlap={train_box_summary.get('max_overlap_area', 0)}, "
            f"r_keep={train_canny_summary.get('r_keep', 1.0):.4f}, "
            f"person_status={person_summary.get('max_status', 'outside')}, "
            f"train_box_in_target={train_box_in_target}, "
            f"bad_template_count={self.bad_template_suspect_count}, "
            f"bad_template_recovery_active={self.bad_template_recovery_active}"
        )

        decision = FrameDecision(
            persons=person_summary.get("people", []),
            # trains=train_box_summary.get("trains", []),
            trains=visual_trains,
            removed_trains=removed_train_dets,
            person_summary=person_summary,
            train_box_summary=train_box_summary,
            train_canny_summary=train_canny_summary,
            train_box_state_raw=train_box_state_raw,
            train_canny_state_raw=train_canny_state_raw,
            train_template_confirmed=train_template_confirmed,
            train_template_reason=train_template_reason,
            train_state_raw=train_state_raw,
            person_state_raw=person_state_raw,
            train_state_stable=train_state_stable,
            person_state_stable=person_state_stable,
            final_state_raw=final_state_raw,
            final_state_stable=final_state_stable,
            sampled=True,
            frame_index=frame_index,
            status_changed=status_changed,
            clear_for_template_update=clear_for_template_update,
            train_box_in_target=train_box_in_target,
            canny_alarm=canny_alarm,
            bad_template_suspect_count=self.bad_template_suspect_count,
            bad_template_recovery_active=self.bad_template_recovery_active,
        )
        self.last_decision = decision
        return decision

    def process_unsampled_frame(self, frame_index: int) -> FrameDecision:
        decision = self.last_decision
        decision.sampled = False
        decision.frame_index = frame_index
        return decision

    def maybe_update_template(self, frame: np.ndarray, decision: FrameDecision) -> None:
        if decision.clear_for_template_update:
            if decision.bad_template_recovery_active:
                reason = "bad_template_recovery_clear_by_boxes_only"
            else:
                reason = "clear_by_boxes_only"
        else:
            reason = "person_or_train_box_detected"

        msg = self.template_updater.maybe_update(
            frame=frame,
            target_clear=decision.clear_for_template_update,
            clear_reason=reason,
            force_rebuild=decision.bad_template_recovery_active,
        )
        if msg:
            log_info(msg)

    def maybe_push_heartbeat(self, decision: FrameDecision) -> None:
        self.status_pusher.maybe_push_heartbeat(decision.final_state_stable)


# ============================================================
# 可视化
# ============================================================
def state_name(code: int) -> str:
    mapping = {
        STATE_CLEAR: "CLEAR",
        STATE_WARNING: "WARNING",
        STATE_TRAIN: "TRAIN",
        STATE_PERSON: "PERSON",
    }
    return mapping.get(int(code), f"UNKNOWN_{code}")


def draw_result(
    frame: np.ndarray,
    target_rail_polygon: np.ndarray,
    mask_polygons: Sequence[np.ndarray],
    decision: FrameDecision,
) -> np.ndarray:
    vis = frame.copy()

    if bool(CONFIG.get("draw_target_polygon", True)):
        cv2.polylines(vis, [target_rail_polygon.reshape((-1, 1, 2))], True, (255, 0, 255), 3)

    if bool(CONFIG.get("draw_mask_polygons", True)):
        for poly in mask_polygons:
            cv2.polylines(vis, [poly.reshape((-1, 1, 2))], True, (128, 128, 128), 2)

    if bool(CONFIG.get("draw_person_boxes", True)):
        for item in decision.persons:
            x1, y1, x2, y2 = item["box_xyxy"]
            status = item["status"]
            if status == "outside":
                color = (0, 180, 0)
            elif status == "near_rail":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"person {float(item['confidence']):.2f} {status}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            if bool(CONFIG.get("draw_person_points", True)):
                for pt_name in ("left_point", "right_point", "mid_point"):
                    px, py = item[pt_name]
                    cv2.circle(vis, (int(px), int(py)), 4, color, -1)

    # if bool(CONFIG.get("draw_train_boxes", True)):
    #     for det in decision.trains:
    #         x1, y1, x2, y2 = det["box_xyxy"]
    #         stage = det.get("overlap_stage", "none")
    #         if stage == "passing":
    #             color = (0, 0, 255)
    #         elif stage == "entering":
    #             color = (0, 255, 255)
    #         else:
    #             color = (255, 255, 0)
    #         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(
    #             vis,
    #             f"train {float(det['confidence']):.2f} overlap={int(det.get('overlap_area', 0))} {stage}",
    #             (x1, max(18, y1 - 8)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.52,
    #             color,
    #             2,
    #         )

    if bool(CONFIG.get("draw_train_boxes", True)):
        for det in decision.trains:
            x1, y1, x2, y2 = det["box_xyxy"]

            draw_state = int(det.get("final_state_code", det.get("state_code", STATE_CLEAR)))
            draw_stage = det.get("final_stage", det.get("overlap_stage", "none"))

            if draw_state == STATE_TRAIN:
                color = (0, 0, 255)  # 红：最终确认有车
            elif draw_state == STATE_WARNING:
                color = (0, 255, 255)  # 黄：最终确认预警
            else:
                color = (255, 255, 0)  # 蓝：最终确认不在目标铁轨

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"train {float(det['confidence']):.2f} overlap={int(det.get('overlap_area', 0))} final={state_name(draw_state)} {draw_stage}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )

    if bool(CONFIG.get("draw_removed_trains", False)):
        for det in decision.removed_trains:
            x1, y1, x2, y2 = det["box_xyxy"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(vis, "removed_train", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    lines = [
        f"frame={decision.frame_index} sampled={int(decision.sampled)}",
        f"person_raw={state_name(decision.person_state_raw)} person_stable={state_name(decision.person_state_stable)} max_status={decision.person_summary.get('max_status', 'outside')}",
        f"train_raw={state_name(decision.train_state_raw)} train_box_raw={state_name(decision.train_box_state_raw)} train_canny_raw={state_name(decision.train_canny_state_raw)}",
        f"train_stable={state_name(decision.train_state_stable)} template_confirmed={int(decision.train_template_confirmed)} overlap={decision.train_box_summary.get('max_overlap_area', 0)} r_keep={decision.train_canny_summary.get('r_keep', 1.0):.4f}",
        f"final_raw={state_name(decision.final_state_raw)} final_stable={state_name(decision.final_state_stable)}",
    ]

    y = 30
    for text in lines:
        cv2.putText(vis, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 2)
        y += 30

    return vis


# ============================================================
# 运行入口
# ============================================================
def run_video_mode() -> None:
    cfg = CONFIG["video"]
    video_path = str(cfg["video_path"])
    output_video_path = str(cfg["output_video_path"])
    log_info(f"[video_mode] input={video_path}, output={output_video_path}")

    cap = open_video_capture(video_path)
    if cap is None:
        raise ValueError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    writer = None
    if bool(CONFIG.get("save_output_video", True)):
        writer = create_video_writer(output_video_path, fps, width, height)

    monitor = RailwayMonitor()
    frame_index = 0
    fps_show = 0.0

    while True:
        t1 = time.time()
        time1 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        sampled = (frame_index % monitor.detect_stride == 0)
        if sampled:
            decision = monitor.process_sampled_frame(
                frame,
                frame_index,
                source_frame_id=frame_index,
                frame_ts=time.time(),
            )
            monitor.maybe_update_template(frame, decision)
        else:
            decision = monitor.process_unsampled_frame(frame_index)

        monitor.maybe_push_heartbeat(decision)
        draw_frame = draw_result(frame, monitor.target_rail_polygon, monitor.mask_polygons, decision)

        cost = time.time() - t1
        if cost > 0:
            fps_show = (fps_show + 1.0 / cost) / 2.0 if fps_show > 0 else (1.0 / cost)
        time2=time.time()
        print(f"整个检测时间{time2-time1}_____________________")
        cv2.putText(draw_frame, f"fps={fps_show:.2f}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)

        if writer is not None:
            writer.write(draw_frame)

        # if bool(CONFIG.get("show_window", False)):
        #     cv2.imshow("railway_monitor", draw_frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key in (27, ord("q")):
        #         break

        if bool(CONFIG.get("show_window", False)):
            # show_frame = cv2.resize(draw_frame, (960, 540), interpolation=cv2.INTER_AREA)
            show_frame = cv2.resize(draw_frame, (1280, 720), interpolation=cv2.INTER_AREA)
            # show_frame = cv2.resize(draw_frame, (1600, 900), interpolation=cv2.INTER_AREA)
            print("video_mode:", draw_frame.shape, "->", show_frame.shape)
            cv2.imshow("railway_monitor", show_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    log_info("视频处理完成")
    if writer is not None:
        log_info(f"输出视频: {output_video_path}")



def run_camera_mode() -> None:
    cfg = CONFIG["camera"]
    camera_name = str(cfg.get("name", "target_rail_camera"))
    rtsp_url = str(cfg["rtsp_url"])
    output_video_path = str(cfg.get("output_video_path", "output/camera_result.mp4"))
    log_info(f"[camera_mode] name={camera_name}, rtsp_url={rtsp_url}, output={output_video_path}")

    reader = RTSPStreamReader(rtsp_url, camera_name=camera_name)
    reader.start()

    monitor = RailwayMonitor()
    writer = None
    frame_index = 0
    fps_show = 0.0
    last_source_frame_id = 0
    last_wait_log_ts = 0.0
    stale_seconds = float(CONFIG.get("frame_stale_seconds", 2.0))
    reconnect_grace = float(CONFIG.get("post_reconnect_grace_seconds", 1.5))
    wait_timeout = min(0.20, max(0.02, float(CONFIG.get("grab_interval", 0.05)) * 2.0))

    # ===== 在这里加：窗口初始化 =====
    # if bool(CONFIG.get("show_window", False)):
    #     cv2.namedWindow("railway_monitor", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("railway_monitor", 640, 360)  # 这里改窗口大小

    try:
        while True:
            t1 = time.time()
            time_begin = datetime.now()
            reader.wait_for_newer_frame(last_source_frame_id, timeout=wait_timeout)
            frame, frame_ts, source_frame_id = reader.get_latest_frame(max_age=stale_seconds)

            if frame is None:
                status = reader.get_status()
                now_ts = time.time()
                if now_ts - last_wait_log_ts >= 2.0:
                    log_warning(
                        f"[camera] 等待最新帧: connection={status['connection_state']}, "
                        f"error_state={status['error_state']}, latest_frame_id={status['latest_frame_id']}, "
                        f"latest_frame_ts={status['latest_frame_ts']:.3f}"
                    )
                    last_wait_log_ts = now_ts

                monitor.maybe_push_heartbeat(monitor.last_decision)
                time.sleep(0.01)
                continue

            if source_frame_id <= last_source_frame_id:
                monitor.maybe_push_heartbeat(monitor.last_decision)
                time.sleep(0.005)
                continue

            last_source_frame_id = source_frame_id

            height, width = frame.shape[:2]
            if writer is None and bool(CONFIG.get("save_output_video", True)):
                writer = create_video_writer(output_video_path, 25.0, width, height)

            status = reader.get_status()
            in_reconnect_grace = bool(
                status["last_reconnect_ts"] and (time.time() - float(status["last_reconnect_ts"]) < reconnect_grace)
            )

            if in_reconnect_grace:
                decision = monitor.process_unsampled_frame(frame_index)
            else:
                sampled = (frame_index % monitor.detect_stride == 0)
                if sampled:
                    decision = monitor.process_sampled_frame(
                        frame,
                        frame_index,
                        source_frame_id=source_frame_id,
                        frame_ts=frame_ts,
                    )
                    monitor.maybe_update_template(frame, decision)
                else:
                    decision = monitor.process_unsampled_frame(frame_index)

            monitor.maybe_push_heartbeat(decision)
            draw_frame = draw_result(frame, monitor.target_rail_polygon, monitor.mask_polygons, decision)

            conn_text = (
                f"conn={status['connection_state']} "
                f"err={status['error_state'] or 'none'} "
                f"src_id={source_frame_id}"
            )
            if in_reconnect_grace:
                conn_text += " reconnect_grace=1"

            time_end = datetime.now()

            cv2.putText(
                draw_frame,
                conn_text,
                (20, max(30, height - 55)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

            cv2.putText(draw_frame, f"start_time{time_begin}", (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 0),
                        2,
                        )
            cv2.putText(draw_frame, f"end_time{time_end}", (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 255, 255),
                        2,
                        )

            cost = time.time() - t1
            if cost > 0:
                fps_show = (fps_show + 1.0 / cost) / 2.0 if fps_show > 0 else (1.0 / cost)

            cv2.putText(
                draw_frame,
                f"fps={fps_show:.2f}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
            )



            if writer is not None:
                writer.write(draw_frame)

            # if bool(CONFIG.get("show_window", False)):
            #     cv2.imshow("railway_monitor", draw_frame)
            #     key = cv2.waitKey(1) & 0xFF
            #     if key in (27, ord("q")):
            #         break

            if bool(CONFIG.get("show_window", False)):
                cv2.namedWindow("railway_monitor", cv2.WINDOW_NORMAL)
                show_frame = cv2.resize(draw_frame, (960, 540), interpolation=cv2.INTER_AREA)
                cv2.imshow("railway_monitor", show_frame)
                cv2.resizeWindow("railway_monitor", 960, 540)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    finally:
        reader.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        log_info("摄像头处理结束")
        if writer is not None:
            log_info(f"输出视频: {output_video_path}")


def main() -> None:
    mode = str(CONFIG.get("input_mode", "video")).lower()
    log_info(
        f"[startup] mode={mode}, backend={CONFIG.get('backend')}, "
        f"log_file={CONFIG.get('log_file_path', 'logs/railway_monitor.log')}"
    )
    if mode == "video":
        run_video_mode()
    elif mode == "camera":
        run_camera_mode()
    else:
        raise ValueError("input_mode 只能是 'video' 或 'camera'")


if __name__ == "__main__":
    main()
