#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_ROSCONSOLE_CFG = os.path.join(_WORKSPACE_ROOT, 'rosconsole_silent_tf.conf')
if os.path.exists(_ROSCONSOLE_CFG) and ('ROSCONSOLE_CONFIG_FILE' not in os.environ):
    os.environ['ROSCONSOLE_CONFIG_FILE'] = _ROSCONSOLE_CFG
# tf2内部重复TF告警来自console_bridge，需单独降级日志级别
os.environ.setdefault('CONSOLE_BRIDGE_LOG_LEVEL', 'error')
os.environ.setdefault('CONSOLE_BRIDGE_log_level', 'error')


class _FilteredConsoleStream(object):
    """Drop known TF repeated-data spam lines for this process only."""

    def __init__(self, raw):
        self._raw = raw
        self._buf = ''

    def write(self, data):
        if not data:
            return 0
        self._buf += data
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if self._drop(line):
                continue
            self._raw.write(line + '\n')
        return len(data)

    def flush(self):
        if self._buf and (not self._drop(self._buf)):
            self._raw.write(self._buf)
        self._buf = ''
        self._raw.flush()

    def _drop(self, line):
        return ('TF_REPEATED_DATA' in line) or ('buffer_core.cpp' in line)


if not isinstance(sys.stderr, _FilteredConsoleStream):
    sys.stderr = _FilteredConsoleStream(sys.stderr)
if not isinstance(sys.stdout, _FilteredConsoleStream):
    sys.stdout = _FilteredConsoleStream(sys.stdout)

import rospy
import numpy as np
import math
import json
import cv2
from datetime import datetime
from collections import deque
import tf2_ros
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from std_msgs.msg import String
try:
    from cv_bridge import CvBridge, CvBridgeError
except Exception:  # cv_bridge may fail to load (ABI/OpenCV mismatch)
    CvBridge = None
    CvBridgeError = Exception
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from tf.transformations import euler_from_quaternion, quaternion_matrix


class MazeSolver:
    def __init__(self):
        rospy.init_node('maze_solver', anonymous=True)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # YOLO探测：相机输入、深度与检测结果输出
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/depth_registered/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/rgb/camera_info')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1)
        self.target_pub = rospy.Publisher('/detected_targets', String, queue_size=10)
        self.bridge = CvBridge() if CvBridge is not None else None
        self._warned_no_bridge = False
        self._warned_no_depth = False

        self.latest_depth_msg = None
        self.camera_fx = None
        self.camera_fy = None
        self.camera_cx = None
        self.camera_cy = None

        # 机器人位姿与激光
        self.current_pos = Point()
        self.current_yaw = 0.0
        self.odom_received = False
        self.scan_msg = None
        self.laser_ranges = []
        self.turn_dir = 1.0
        self.rate = rospy.Rate(10)

        # 控制参数
        self.speed_scale = float(rospy.get_param('~speed_scale', 1.0))
        
        # ===== 改进的避障参数 =====
        # 最小安全距离 = 机器人半径 + 安全间隙 + 额外缓冲
        # 由于有地图虚拟传感器双重保护，可以适当降低警惕半径，避免过度保守导致绕圈
        self.min_safe_distance = float(rospy.get_param('~min_safe_distance', 0.42))  # 降低到0.42m（从0.55m）
        self.safe_distance = self.min_safe_distance  # 触发避障的距离
        self.danger_distance = 0.30  # 危险距离，立即停止（从0.35m降低）
        self.critical_distance = 0.20  # 临界距离，急速后退（从0.25m降低）
        
        # 与官方DWA参数对齐：可通过launch直接覆盖。
        self.max_linear = float(rospy.get_param('~max_linear', 0.30 * self.speed_scale))
        self.max_angular = float(rospy.get_param('~max_angular', 1.2 * self.speed_scale))
        self.goal_tolerance = 0.28
        self.yaw_tolerance = 0.35

        # 无墙跟随的简化避障
        # （不再依赖沿墙跟随，改为动态窗口法则）
        self.wall_dist = 0.60  # 保持距离（不贴墙）
        self.wall_kp = 0.8  # 降低增益，避免过度反应
        self.wall_linear = 0.12 * self.speed_scale  # 降低速度

        # 简化的导航状态（不再使用复杂的Bug状态机）
        # nav_state 仅保留用于某些日志/控制流，但不是主要决策依据

        # 路径与frontier状态
        self.current_frontier = None
        self.path_waypoints = []
        self.wp_index = 0
        self.waypoint_step = 3
        self.waypoint_reach_tol = 0.22
        self.replan_interval = 2.0
        self.last_plan_time = 0.0
        self.min_path_cells = 10
        self.min_frontier_goal_dist = float(rospy.get_param('~min_frontier_goal_dist', 0.70))

        # frontier选取参数
        self.frontier_min_cluster = 6
        self.frontier_blacklist = deque(maxlen=100)  # (x, y, ts)
        self.blacklist_radius = 1.5
        self.blacklist_ttl_sec = 60.0
        self.recent_frontiers = deque(maxlen=20)  # (x, y)
        self.revisit_radius = float(rospy.get_param('~revisit_radius', 1.3))
        self.last_forced_switch_time = 0.0
        self.forced_switch_cooldown = 3.0
        self.force_escape_until = 0.0
        self.obstacle_escape_dir = 0.0
        self.obstacle_escape_until = 0.0
        self.obstacle_clear_streak = 0
        self.obstacle_clear_needed = int(rospy.get_param('~obstacle_clear_needed', 4))
        self.obstacle_escape_hold_sec = float(rospy.get_param('~obstacle_escape_hold_sec', 1.0))
        # 逃逸释放条件也相应放宽，避免陷入长期逃逸状态
        self.escape_min_forward_clear = float(rospy.get_param('~escape_min_forward_clear', 0.58))  # 从0.72m降低
        self.escape_min_side_clear = float(rospy.get_param('~escape_min_side_clear', 0.54))  # 从0.68m降低

        self.frontier_unknown_window_cells = int(rospy.get_param('~frontier_unknown_window_cells', 6))
        self.frontier_size_weight = float(rospy.get_param('~frontier_size_weight', 2.0))
        self.frontier_gain_weight = float(rospy.get_param('~frontier_gain_weight', 0.85))
        self.frontier_dist_weight = float(rospy.get_param('~frontier_dist_weight', 0.16))
        self.frontier_revisit_penalty = float(rospy.get_param('~frontier_revisit_penalty', 14.0))
        self.frontier_min_unknown_gain = int(rospy.get_param('~frontier_min_unknown_gain', 8))

        self.nav_log_interval_sec = float(rospy.get_param('~nav_log_interval_sec', 1.5))
        self._last_nav_log_tag = ''
        self._last_nav_log_time = 0.0
        self._nav_log_silence = {
            'nav_tracking',
            'stop:到达当前路点，等待下一个目标',
        }

        self.robot_radius_m = 0.18
        # 真实环境不宜过度膨胀，否则会把可通行通道压窄；这里适当收紧缓冲层。
        self.clearance_m = float(rospy.get_param('~clearance_m', 0.08))
        self.inflation_radius_m = float(rospy.get_param('~inflation_radius_m', self.robot_radius_m + self.clearance_m + 0.04))
        self.occ_threshold = int(rospy.get_param('~occ_threshold', 30))
        self._inflate_radius_cells = -1
        self._inflate_offsets = []

        # 完成判定：连续多次无未知栅格才停
        self.no_unknown_count = 0
        self.no_unknown_need = 20  # 10Hz下约2秒

        # 卡死检测与恢复
        self._pose_hist = deque(maxlen=40)
        self._stuck_dist_eps = 0.03
        self._stuck_time_sec = 6.0
        self._recovery_until = 0.0
        self._stuck_event_count = 0
        self.max_stuck_before_replan = 2

        # 局部最优脱困模式
        self.local_minima_until = 0.0
        self.local_minima_count = 0

        # 进展监测：长时间无位移则强制换frontier
        self._last_progress_pose = None
        self._last_progress_time = 0.0
        self.progress_eps = 0.06
        self.no_progress_timeout = 8.0

        # 地图
        self.map_data = None
        self.map_info = None
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # TF：优先使用map坐标系位姿做规划
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_pose_valid = False
        self.map_pos_x = 0.0
        self.map_pos_y = 0.0
        self.map_yaw = 0.0

        # ========== YOLO探测配置 ==========
        self.enable_color_detection = bool(rospy.get_param('~enable_color_detection', True))
        self.enable_yolo_detection = bool(rospy.get_param('~enable_yolo_detection', False))
        self.yolo_model_path = rospy.get_param('~yolo_model_path', 'yolov8n.pt')
        self.yolo_conf_thresh = float(rospy.get_param('~yolo_conf_thresh', 0.40))
        self.yolo_iou_thresh = float(rospy.get_param('~yolo_iou_thresh', 0.45))
        self.yolo_imgsz = int(rospy.get_param('~yolo_imgsz', 640))
        self.yolo_infer_interval_sec = float(rospy.get_param('~yolo_infer_interval_sec', 0.20))
        self.min_bbox_area = float(rospy.get_param('~min_bbox_area', 250.0))
        self.max_depth_m = float(rospy.get_param('~max_depth_m', 5.0))
        self._last_yolo_infer_time = 0.0
        self._warned_no_yolo = False

        self.yolo_model = None
        self.yolo_class_names = {}
        if self.enable_yolo_detection:
            if YOLO is None:
                rospy.logerr('YOLOv8未安装：请先安装 ultralytics 后再运行。')
            else:
                try:
                    self.yolo_model = YOLO(self.yolo_model_path)
                    names = getattr(self.yolo_model, 'names', {})
                    if isinstance(names, dict):
                        self.yolo_class_names = {int(k): str(v) for k, v in names.items()}
                    rospy.loginfo('YOLOv8模型已加载: %s', self.yolo_model_path)
                except Exception as e:
                    rospy.logerr('YOLOv8模型加载失败(%s): %s', self.yolo_model_path, e)

        self.detection_merge_radius = float(rospy.get_param('~detection_merge_radius', 0.45))
        default_report_dir = os.path.join(_WORKSPACE_ROOT, 'src', 'patrol_task')
        self.report_dir = rospy.get_param('~report_dir', default_report_dir)
        run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_json = os.path.join(self.report_dir, 'patrol_report_%s.json' % run_tag)
        default_md = os.path.join(self.report_dir, 'patrol_report_%s.md' % run_tag)
        self.report_output_path = rospy.get_param('~report_output_path', default_json)
        self.report_autosave_sec = float(rospy.get_param('~report_autosave_sec', 5.0))
        self._last_report_save_time = 0.0
        self.target_z = float(rospy.get_param('~target_z', 0.1))
        self._final_report_printed = False
        self.markdown_report_path = rospy.get_param('~markdown_report_path', default_md)
        self.enable_patrol_report = bool(rospy.get_param('~enable_patrol_report', False))

        # 返航触发：未知区域长时间变化很小（B方案）
        self.unknown_stable_window_sec = float(rospy.get_param('~unknown_stable_window_sec', 120.0))
        self.unknown_stable_min_runtime_sec = float(rospy.get_param('~unknown_stable_min_runtime_sec', 90.0))
        self.unknown_stable_delta_cells = int(rospy.get_param('~unknown_stable_delta_cells', 60))
        self.unknown_history = deque(maxlen=3000)  # (time, unknown_cells)
        self.start_time = rospy.get_time()
        self.return_mode = False
        self.return_completed = False
        self.force_return_time_sec = float(rospy.get_param('~force_return_time_sec', 105.0))
        self._force_return_logged = False
        self.frontiers_exhausted = False
        self.frontier_missing_since = 0.0
        self.return_requires_no_frontier_sec = float(rospy.get_param('~return_requires_no_frontier_sec', 45.0))

        # 起点位姿（A方案：程序启动后的初始位姿）
        self.home_pose = None  # (x, y, yaw)
        self.home_reach_tol = float(rospy.get_param('~home_reach_tol', 0.25))
        self.home_yaw_tol = float(rospy.get_param('~home_yaw_tol', 0.20))

        # 简化避障所需的参数（已集成到navigate_bug_to_target的新逻辑中）

        # 闭合回路返航判定（D方案）：轨迹闭环 + 包围框已探明比例高
        self.enable_loop_return = bool(rospy.get_param('~enable_loop_return', True))
        self.loop_close_dist = float(rospy.get_param('~loop_close_dist', 0.35))
        self.loop_min_path_len = float(rospy.get_param('~loop_min_path_len', 8.0))
        self.loop_min_points_gap = int(rospy.get_param('~loop_min_points_gap', 25))
        self.loop_known_ratio_thresh = float(rospy.get_param('~loop_known_ratio_thresh', 0.86))
        self.loop_bbox_min_area = float(rospy.get_param('~loop_bbox_min_area', 6.0))
        self.loop_no_frontier_sec = float(rospy.get_param('~loop_no_frontier_sec', 20.0))
        self.loop_pose_hist = deque(maxlen=2400)  # (x, y)
        self.loop_cumlen_hist = deque(maxlen=2400)  # cumulative length at sample
        self.loop_path_total = 0.0
        self.loop_sample_step = float(rospy.get_param('~loop_sample_step', 0.10))
        self._loop_return_logged = False
        self._coverage_return_logged = False

        self.return_known_ratio_thresh = float(rospy.get_param('~return_known_ratio_thresh', 0.965))
        self.return_unknown_cells_thresh = int(rospy.get_param('~return_unknown_cells_thresh', 120))
        self.return_frontier_missing_sec = float(rospy.get_param('~return_frontier_missing_sec', 10.0))
        self.return_min_runtime_sec = float(rospy.get_param('~return_min_runtime_sec', 30.0))

        # 聚类参数（按类别分别做空间聚类）
        # 大幅降低聚类半径，使得相近的点更容易被合并成同一物体
        # 实际只有3-5个物体，不能聚类到20+个，这说明参数还要继续降低
        self.cluster_radius = float(rospy.get_param('~cluster_radius', 0.45))  # 从0.65改为0.45（更紧凑）
        self.cluster_merge_radius = float(rospy.get_param('~cluster_merge_radius', 0.55))  # 从0.85改为0.55（更激进合并）
        self.cluster_max_span = float(rospy.get_param('~cluster_max_span', 0.50))  # 从0.75改为0.50（限制簇跨度）

        # 新目标确认机制：连续命中后再入库，减少单帧误检导致的假目标
        self.pending_confirm_hits = int(rospy.get_param('~pending_confirm_hits', 2))
        self.pending_ttl_sec = float(rospy.get_param('~pending_ttl_sec', 2.5))
        # 进一步降低pending合并半径，在确认前就把相近点更激进地合并
        self.pending_merge_radius = float(rospy.get_param('~pending_merge_radius', 0.30))  # 从0.40改为0.30（激进合并）
        
        # 检测阶段的合并半径（当新检测点接近已有目标时，直接合并而不进入pending）
        self.detection_merge_radius = float(rospy.get_param('~detection_merge_radius', 0.35))  # 从0.45改为0.35
        self.pending_targets = {}

        # 按类别记录目标：{label: [{x,y,ts,px,py,area}, ...]}
        self.detected_targets = {}
        # 按类别记录“确认后的全部探测事件”（不做去重，用于追溯）
        self.detection_events = {}
        # 追踪每种颜色是否已在终端输出过首次确认，用于去重终端日志
        self.color_first_logged = {}
        self.event_assoc_radius = float(rospy.get_param('~event_assoc_radius', 0.9))

        rospy.on_shutdown(self._on_shutdown)

    def image_callback(self, msg):
        if not self.enable_color_detection and not self.enable_yolo_detection:
            return

        if self.enable_color_detection and self.latest_depth_msg is not None and (self.camera_fx is not None) and (self.camera_fy is not None) and (self.camera_cx is not None) and (self.camera_cy is not None):
            if self._detect_color_targets(msg):
                return

        if not self.enable_yolo_detection:
            return

        if self.yolo_model is None:
            if not self._warned_no_yolo:
                rospy.logwarn('YOLOv8未就绪，跳过图像检测。')
                self._warned_no_yolo = True
            return

        if (self.camera_fx is None) or (self.camera_fy is None) or (self.camera_cx is None) or (self.camera_cy is None):
            rospy.logwarn_throttle(2.0, '等待相机内参/camera_info...')
            return

        if self.latest_depth_msg is None:
            if not self._warned_no_depth:
                rospy.logwarn('等待深度图像: %s', self.depth_topic)
                self._warned_no_depth = True
            return

        now = rospy.get_time()
        if (now - self._last_yolo_infer_time) < self.yolo_infer_interval_sec:
            return
        self._last_yolo_infer_time = now

        frame_rgb = self._image_msg_to_rgb(msg)
        if frame_rgb is None or frame_rgb.size == 0:
            return

        depth_m = self._depth_msg_to_meters(self.latest_depth_msg)
        if depth_m is None:
            rospy.logwarn_throttle(2.0, '深度图解码失败，跳过本帧YOLO定位。')
            return

        result = self._predict_yolo(frame_rgb)
        if result is None:
            return

        found_any = False
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) <= 0:
            return

        depth_frame = str(self.latest_depth_msg.header.frame_id) if self.latest_depth_msg.header.frame_id else 'camera_link'
        depth_stamp = self.latest_depth_msg.header.stamp if self.latest_depth_msg.header.stamp else rospy.Time(0)

        for box in boxes:
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.yolo_conf_thresh:
                    continue

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                if area < self.min_bbox_area:
                    continue

                px = int(round((x1 + x2) * 0.5))
                py = int(round((y1 + y2) * 0.5))

                z = self._depth_at_pixel(depth_m, px, py)
                if (not math.isfinite(z)) or z <= 0.0 or z > self.max_depth_m:
                    continue

                x_cam = (float(px) - self.camera_cx) * z / self.camera_fx
                y_cam = (float(py) - self.camera_cy) * z / self.camera_fy

                map_point = self._point_camera_to_map(x_cam, y_cam, z, depth_frame, depth_stamp)
                if map_point is None:
                    continue

                tx, ty, tz = map_point
                label = self.yolo_class_names.get(cls_id, str(cls_id))
                if self._register_target(label, tx, ty, px, py, area, tz):
                    found_any = True
                    rospy.loginfo_throttle(1.0, 'YOLO目标: %s conf=%.2f map=(%.2f, %.2f, %.2f)',
                                           label, conf, tx, ty, tz)
            except Exception:
                continue

        if found_any:
            self._publish_detection_overview()

    def _detect_color_targets(self, msg):
        frame_rgb = self._image_msg_to_rgb(msg)
        if frame_rgb is None or frame_rgb.size == 0:
            return False

        depth_m = self._depth_msg_to_meters(self.latest_depth_msg)
        if depth_m is None:
            return False

        depth_frame = str(self.latest_depth_msg.header.frame_id) if self.latest_depth_msg.header.frame_id else 'camera_link'
        depth_stamp = self.latest_depth_msg.header.stamp if self.latest_depth_msg.header.stamp else rospy.Time(0)

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        color_ranges = {
            'red1': (np.array([0, 120, 70]), np.array([10, 255, 255])),
            'red2': (np.array([170, 120, 70]), np.array([180, 255, 255])),
            'green': (np.array([35, 100, 100]), np.array([85, 255, 255])),
            'blue': (np.array([100, 150, 0]), np.array([140, 255, 255])),
            'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
        }

        found_any = False
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            area_threshold = max(float(self.min_bbox_area), 1500.0)
            if area < area_threshold:
                continue

            moments = cv2.moments(contour)
            if moments.get('m00', 0.0) == 0.0:
                continue

            px = int(round(moments['m10'] / moments['m00']))
            py = int(round(moments['m01'] / moments['m00']))
            z = self._depth_at_pixel(depth_m, px, py)
            if (not math.isfinite(z)) or z <= 0.0 or z > self.max_depth_m:
                continue

            x_cam = (float(px) - self.camera_cx) * z / self.camera_fx
            y_cam = (float(py) - self.camera_cy) * z / self.camera_fy
            map_point = self._point_camera_to_map(x_cam, y_cam, z, depth_frame, depth_stamp)
            if map_point is None:
                continue

            tx, ty, tz = map_point
            display_name = 'red' if color_name.startswith('red') else color_name
            # 不在检测阶段输出日志，只在首次确认时输出（见_register_target中的color_first_logged）
            if self._register_target(display_name, tx, ty, px, py, area, tz):
                found_any = True

        if found_any:
            self._publish_detection_overview()
        return found_any

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def camera_info_callback(self, msg):
        if msg is None or len(msg.K) < 9:
            return
        self.camera_fx = float(msg.K[0])
        self.camera_fy = float(msg.K[4])
        self.camera_cx = float(msg.K[2])
        self.camera_cy = float(msg.K[5])

    def _predict_yolo(self, frame_rgb):
        try:
            results = self.yolo_model.predict(
                source=frame_rgb,
                conf=self.yolo_conf_thresh,
                iou=self.yolo_iou_thresh,
                imgsz=self.yolo_imgsz,
                verbose=False,
            )
            if not results:
                return None
            return results[0]
        except Exception as e:
            rospy.logwarn_throttle(2.0, 'YOLO推理失败: %s', e)
            return None

    def _image_msg_to_rgb(self, msg):
        """Convert sensor_msgs/Image to RGB ndarray, without requiring cv2."""
        # Preferred path: cv_bridge (if available)
        if self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            except CvBridgeError:
                pass
            except Exception:
                pass

        # Manual fallback for common raw image encodings
        try:
            h = int(msg.height)
            w = int(msg.width)
            if h <= 0 or w <= 0:
                return None

            enc = (msg.encoding or '').lower()
            buf = np.frombuffer(msg.data, dtype=np.uint8)

            if enc == 'rgb8':
                expected = h * w * 3
                if buf.size < expected:
                    return None
                return buf[:expected].reshape((h, w, 3))

            if enc == 'bgr8':
                expected = h * w * 3
                if buf.size < expected:
                    return None
                bgr = buf[:expected].reshape((h, w, 3))
                return bgr[:, :, ::-1]

            if enc == 'mono8':
                expected = h * w
                if buf.size < expected:
                    return None
                gray = buf[:expected].reshape((h, w))
                return np.repeat(gray[:, :, None], 3, axis=2)

            # Unknown encoding
            if not self._warned_no_bridge:
                rospy.logwarn('无法解码图像编码: %s (cv_bridge不可用或转换失败)', msg.encoding)
                self._warned_no_bridge = True
            return None
        except Exception:
            return None

    def _depth_msg_to_meters(self, msg):
        """Convert depth image to float32 meters."""
        if msg is None:
            return None

        if self.bridge is not None:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if depth is None:
                    return None
                if depth.dtype == np.uint16:
                    return depth.astype(np.float32) / 1000.0
                return depth.astype(np.float32)
            except Exception:
                pass

        try:
            h = int(msg.height)
            w = int(msg.width)
            if h <= 0 or w <= 0:
                return None
            enc = (msg.encoding or '').lower()
            buf = np.frombuffer(msg.data, dtype=np.uint8)

            if enc in ('16uc1', 'mono16'):
                expected = h * w * 2
                if buf.size < expected:
                    return None
                depth_u16 = buf[:expected].view(np.uint16).reshape((h, w))
                return depth_u16.astype(np.float32) / 1000.0

            if enc in ('32fc1', '32fc'):
                expected = h * w * 4
                if buf.size < expected:
                    return None
                return buf[:expected].view(np.float32).reshape((h, w))
        except Exception:
            return None

        return None

    def _depth_at_pixel(self, depth_m, px, py):
        h, w = depth_m.shape[:2]
        if w <= 0 or h <= 0:
            return float('nan')

        x = int(max(0, min(w - 1, int(px))))
        y = int(max(0, min(h - 1, int(py))))
        z = float(depth_m[y, x])
        if math.isfinite(z) and z > 0.0:
            return z

        # 中心点无效时，用邻域中位数补值，提升真实场景鲁棒性
        x0 = max(0, x - 2)
        x1 = min(w, x + 3)
        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        patch = depth_m[y0:y1, x0:x1].reshape(-1)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size <= 0:
            return float('nan')
        return float(np.median(valid))

    def _point_camera_to_map(self, x, y, z, src_frame, stamp):
        if not src_frame:
            return None

        trans = None
        try:
            trans = self.tf_buffer.lookup_transform('map', src_frame, stamp, rospy.Duration(0.05))
        except Exception:
            try:
                trans = self.tf_buffer.lookup_transform('map', src_frame, rospy.Time(0), rospy.Duration(0.05))
            except Exception:
                return None

        q = trans.transform.rotation
        t = trans.transform.translation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        p_cam = np.array([float(x), float(y), float(z)], dtype=np.float64)
        p_map = np.dot(rot, p_cam)
        return (float(p_map[0] + t.x), float(p_map[1] + t.y), float(p_map[2] + t.z))

    def _build_color_mask_numpy(self, frame_rgb, color_name):
        """Simple robust color masks in RGB space for Gazebo-like scenes."""
        r = frame_rgb[:, :, 0].astype(np.int16)
        g = frame_rgb[:, :, 1].astype(np.int16)
        b = frame_rgb[:, :, 2].astype(np.int16)

        if color_name == 'red':
            return (r > 120) & (r > g + 35) & (r > b + 35)
        if color_name == 'green':
            return (g > 110) & (g > r + 25) & (g > b + 25)
        if color_name == 'blue':
            return (b > 100) & (b > r + 25) & (b > g + 25)
        if color_name == 'yellow':
            return (r > 120) & (g > 120) & (b < 110) & (np.abs(r - g) < 70)
        return None

    def _largest_component_stats(self, mask):
        """Return centroid (cx, cy) and area of the largest 8-connected component."""
        if mask is None:
            return 0.0, 0.0, 0.0

        h, w = mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return 0.0, 0.0, 0.0

        best_area = 0
        best_sum_x = 0.0
        best_sum_y = 0.0

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for x0, y0 in zip(xs, ys):
            if visited[y0, x0]:
                continue
            if not mask[y0, x0]:
                continue

            q = deque()
            q.append((x0, y0))
            visited[y0, x0] = 1

            area = 0
            sum_x = 0.0
            sum_y = 0.0

            while q:
                x, y = q.popleft()
                area += 1
                sum_x += float(x)
                sum_y += float(y)

                for dx, dy in neighbors:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx] or (not mask[ny, nx]):
                        continue
                    visited[ny, nx] = 1
                    q.append((nx, ny))

            if area > best_area:
                best_area = area
                best_sum_x = sum_x
                best_sum_y = sum_y

        if best_area <= 0:
            return 0.0, 0.0, 0.0

        return (best_sum_x / best_area), (best_sum_y / best_area), float(best_area)

    def _ensure_label_bucket(self, label_name):
        label = str(label_name)
        if label not in self.pending_targets:
            self.pending_targets[label] = []
        if label not in self.detected_targets:
            self.detected_targets[label] = []
        if label not in self.detection_events:
            self.detection_events[label] = []
        return label

    def _register_target(self, color_name, x, y, px, py, area, z=None):
        color_name = self._ensure_label_bucket(color_name)
        if z is None:
            z = self.target_z
        now = rospy.get_time()
        self._prune_pending(color_name, now)

        target_list = self.detected_targets[color_name]
        for item in target_list:
            if math.hypot(item['x'] - x, item['y'] - y) < self.detection_merge_radius:
                # 已有目标则轻量更新位置（指数平滑）
                alpha = 0.35
                item['x'] = alpha * x + (1.0 - alpha) * item['x']
                item['y'] = alpha * y + (1.0 - alpha) * item['y']
                item['z'] = alpha * float(z) + (1.0 - alpha) * float(item.get('z', self.target_z))
                item['ts'] = rospy.get_time()
                item['px'] = float(px)
                item['py'] = float(py)
                item['area'] = float(area)
                self._append_detection_event(color_name, item['x'], item['y'], item['z'], px, py, area, now)
                # 不输出目标更新日志，避免重复刷屏
                return False

        # 新点先进入pending池，连续命中后再确认，抑制单帧噪声/误色
        pending = self.pending_targets[color_name]
        for p in pending:
            if math.hypot(p['x'] - x, p['y'] - y) < self.pending_merge_radius:
                beta = 0.45
                p['x'] = beta * float(x) + (1.0 - beta) * float(p['x'])
                p['y'] = beta * float(y) + (1.0 - beta) * float(p['y'])
                p['px'] = float(px)
                p['py'] = float(py)
                p['area'] = float(area)
                p['z'] = beta * float(z) + (1.0 - beta) * float(p.get('z', self.target_z))
                p['hits'] = int(p['hits']) + 1
                p['ts'] = now
                if p['hits'] < self.pending_confirm_hits:
                    return False

                x = p['x']
                y = p['y']
                z = p.get('z', z)
                area = p['area']
                pending.remove(p)
                break
        else:
            pending.append({
                'x': float(x),
                'y': float(y),
                'px': float(px),
                'py': float(py),
                'area': float(area),
                'z': float(z),
                'hits': 1,
                'ts': now,
            })
            return False

        target_list.append({
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'color': str(color_name),
            'ts': now,
            'px': float(px),
            'py': float(py),
            'area': float(area),
        })
        self._append_detection_event(color_name, x, y, z, px, py, area, now)
        
        # 每种颜色只在首次确认时在终端输出，后续只记录不输出（避免刷屏）
        if color_name not in self.color_first_logged:
            self.color_first_logged[color_name] = True
            rospy.loginfo('🎯 【首次确认】检测到%s物体 坐标: (%.2f, %.2f, %.2f)', color_name, x, y, z)
        # 后续再遇到同色物体只记录到detected_targets，不输出到终端
        
        return True

    def _append_detection_event(self, color_name, x, y, z, px, py, area, ts):
        color_name = self._ensure_label_bucket(color_name)
        self.detection_events[color_name].append({
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'color': str(color_name),
            'ts': float(ts),
            'px': float(px),
            'py': float(py),
            'area': float(area),
        })

    def _prune_pending(self, color_name, now):
        color_name = self._ensure_label_bucket(color_name)
        pending = self.pending_targets[color_name]
        keep = []
        for p in pending:
            if (now - float(p.get('ts', 0.0))) <= self.pending_ttl_sec:
                keep.append(p)
        self.pending_targets[color_name] = keep

    def _publish_detection_overview(self):
        # 巡逻报告阶段性暂停，仅发布轻量检测统计。
        overview = {
            'stamp': rospy.get_time(),
            'mode': 'yolov8',
            'detected_summary': {k: len(v) for k, v in self.detected_targets.items()},
            'total_detected': int(sum(len(v) for v in self.detected_targets.values())),
        }
        try:
            self.target_pub.publish(String(data=json.dumps(overview, ensure_ascii=False)))
        except Exception:
            pass

    def _publish_motion(self, twist):
        # 统一限幅发布；速度倍率已在控制参数初始化时生效，避免重复放大
        cmd = Twist()
        cmd.linear.x = max(-self.max_linear, min(self.max_linear, float(twist.linear.x)))
        cmd.angular.z = max(-self.max_angular, min(self.max_angular, float(twist.angular.z)))
        self.vel_pub.publish(cmd)

    def _log_nav_status(self, tag, msg):
        now = rospy.get_time()
        if tag in self._nav_log_silence:
            return
        if (tag != self._last_nav_log_tag) or ((now - self._last_nav_log_time) >= self.nav_log_interval_sec):
            rospy.loginfo(msg)
            self._last_nav_log_tag = str(tag)
            self._last_nav_log_time = now

    def _publish_stop(self, reason):
        self._log_nav_status('stop:' + str(reason), '短暂停顿: %s' % reason)
        self._publish_motion(Twist())

    def _finalize_patrol_report(self):
        if self._final_report_printed:
            return
        try:
            report = self._build_target_report()
            self._save_report_to_file(report)
            self._print_final_report(report)
            
            # 输出报告生成完成提示
            raw_summary = report.get('raw_summary', {})
            cluster_summary = report.get('summary', {})
            rospy.loginfo('\n==================== 巡逻报告已生成 ====================')
            rospy.loginfo('JSON报告已保存: %s', self.report_output_path)
            rospy.loginfo('Markdown报告已保存: %s', self.markdown_report_path)
            rospy.loginfo('检测到的目标统计:')
            for color in ('red', 'green', 'yellow', 'blue'):
                count = raw_summary.get(color, 0)
                if count > 0:
                    rospy.loginfo('  - %s: %d个', color, count)
            rospy.loginfo('聚类后的目标数: %d', int(cluster_summary.get('total', 0)))
            rospy.loginfo('='*50 + '\n')
        except Exception as e:
            rospy.logwarn('生成巡逻报告失败: %s', e)

    def _should_start_return_by_coverage(self):
        if self.map_data is None:
            return False

        now = rospy.get_time()
        known_ratio = self._map_known_ratio()
        unknown_cells = self._map_unknown_cells()
        if known_ratio < self.return_known_ratio_thresh:
            return False
        if unknown_cells > self.return_unknown_cells_thresh:
            return False

        if (now - self.start_time) < self.return_min_runtime_sec:
            return False

        if self.frontier_missing_since <= 0.0:
            # 覆盖率已很高且运行足够久时允许直接返航
            if (now - self.start_time) >= self.unknown_stable_min_runtime_sec:
                if not self._coverage_return_logged:
                    rospy.loginfo('覆盖率触发返航: known_ratio=%.3f unknown_cells=%d', known_ratio, unknown_cells)
                    self._coverage_return_logged = True
                return True
            self.frontier_missing_since = now
            return False

        if (now - self.frontier_missing_since) < self.return_frontier_missing_sec:
            return False

        if not self._coverage_return_logged:
            rospy.loginfo('覆盖率触发返航: known_ratio=%.3f unknown_cells=%d', known_ratio, unknown_cells)
            self._coverage_return_logged = True
        return True

    def _map_known_ratio(self):
        if self.map_data is None:
            return 0.0
        known = int(np.count_nonzero(self.map_data != -1))
        total = int(self.map_data.size)
        if total <= 0:
            return 0.0
        return float(known) / float(total)

    def _map_unknown_cells(self):
        if self.map_data is None:
            return 0
        return int(np.count_nonzero(self.map_data == -1))

    def _publish_target_report(self):
        report = self._build_target_report()
        self.target_pub.publish(String(data=json.dumps(report, ensure_ascii=False)))

        now = rospy.get_time()
        if (now - self._last_report_save_time) >= self.report_autosave_sec:
            self._save_report_to_file(report)
            self._last_report_save_time = now

    def _build_target_report(self):
        clusters = self._cluster_all_targets()
        self._attach_all_detection_events(clusters)
        cluster_summary = {}
        for c in clusters:
            color = c['color']
            cluster_summary[color] = int(cluster_summary.get(color, 0)) + 1
        for legacy in ('red', 'green', 'yellow', 'blue'):
            cluster_summary.setdefault(legacy, 0)
        cluster_summary['total'] = len(clusters)

        raw_summary = {k: len(v) for k, v in self.detected_targets.items()}
        all_detection_summary = {k: len(v) for k, v in self.detection_events.items()}
        for legacy in ('red', 'green', 'yellow', 'blue'):
            raw_summary.setdefault(legacy, 0)
            all_detection_summary.setdefault(legacy, 0)
        all_detection_summary['total'] = int(sum(v for k, v in all_detection_summary.items() if k != 'total'))

        return {
            'stamp': rospy.get_time(),
            'raw_summary': raw_summary,
            'all_detection_summary': all_detection_summary,
            'summary': cluster_summary,
            'targets_raw': self.detected_targets,
            'targets_clustered': clusters,
            'frame': 'map_or_odom_fallback',
        }

    def _attach_all_detection_events(self, clusters):
        if not clusters:
            return

        for c in clusters:
            c['detections_all'] = []
            c['detections_all_count'] = 0

        by_color = {}
        for idx, c in enumerate(clusters):
            color = c.get('color', 'unknown')
            by_color.setdefault(color, []).append((idx, c))

        for color, events in self.detection_events.items():
            cands = by_color.get(color, [])
            if not cands:
                continue

            for e in events:
                ex = float(e.get('x', 0.0))
                ey = float(e.get('y', 0.0))
                best_idx = -1
                best_dist = 1e18

                for idx, c in cands:
                    cx = float(c.get('x', 0.0))
                    cy = float(c.get('y', 0.0))
                    span = self._cluster_span_radius(c.get('members', []))
                    radius = max(self.event_assoc_radius, span + 0.35)
                    d = math.hypot(ex - cx, ey - cy)
                    if d <= radius and d < best_dist:
                        best_dist = d
                        best_idx = idx

                if best_idx >= 0:
                    clusters[best_idx]['detections_all'].append(e)

        for c in clusters:
            c['detections_all'].sort(key=lambda x: float(x.get('ts', 0.0)))
            c['detections_all_count'] = int(len(c['detections_all']))

    def _save_report_to_file(self, report):
        try:
            out_dir = os.path.dirname(self.report_output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.report_output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 验证报告内容并输出统计
            targets_raw = report.get('targets_raw', {})
            detected_count = sum(len(v) for v in targets_raw.values())
            rospy.loginfo('报告已写入: %s (检测到%d个独立目标)', self.report_output_path, detected_count)

            self._save_markdown_report(report)
        except Exception as e:
            rospy.logwarn_throttle(5.0, '写入探测报告失败: %s', e)

    def _save_markdown_report(self, report):
        try:
            out_dir = os.path.dirname(self.markdown_report_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            clusters = report.get('targets_clustered', [])
            stamp = float(report.get('stamp', rospy.get_time()))
            time_text = datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S')
            summary = report.get('summary', {})
            max_members_show = 5
            max_events_show = 5
            max_raw_show = 5

            lines = []
            lines.append('# Patrol Report')
            lines.append('')
            lines.append('- Generated at: %s' % time_text)
            lines.append('- Total clustered targets: %d' % int(summary.get('total', len(clusters))))
            lines.append('- By color: red=%d, green=%d, yellow=%d, blue=%d' % (
                int(summary.get('red', 0)),
                int(summary.get('green', 0)),
                int(summary.get('yellow', 0)),
                int(summary.get('blue', 0)),
            ))
            all_sum = report.get('all_detection_summary', {})
            lines.append('- Confirmed detection events: total=%d (red=%d, green=%d, yellow=%d, blue=%d)' % (
                int(all_sum.get('total', 0)),
                int(all_sum.get('red', 0)),
                int(all_sum.get('green', 0)),
                int(all_sum.get('yellow', 0)),
                int(all_sum.get('blue', 0)),
            ))
            lines.append('')
            # =============== 物体详细信息（按聚类）===============
            lines.append('## Objects (Clustered Targets)')
            lines.append('')
            for c in clusters:
                cid = int(c.get('id', 0))
                color = str(c.get('color', 'unknown'))
                cx = float(c.get('x', 0.0))
                cy = float(c.get('y', 0.0))
                cz = float(c.get('z', self.target_z))
                cnt = int(c.get('points_count', 0))
                lines.append('### Object %d' % cid)
                lines.append('- Color: %s' % color)
                lines.append('- Center: (%.2f, %.2f, %.2f)' % (cx, cy, cz))
                lines.append('- Points: %d' % cnt)
                lines.append('- Raw detections:')
                members = c.get('members', [])
                for i, m in enumerate(members[:max_members_show], 1):
                    lines.append('  - %d) (%.2f, %.2f, %.2f)' % (
                        i,
                        float(m.get('x', 0.0)),
                        float(m.get('y', 0.0)),
                        float(m.get('z', self.target_z)),
                    ))
                if len(members) > max_members_show:
                    lines.append('  - ... (%d more)' % (len(members) - max_members_show))
                lines.append('- All confirmed detection events: %d' % int(c.get('detections_all_count', 0)))
                all_events = c.get('detections_all', [])
                for i, m in enumerate(all_events[:max_events_show], 1):
                    lines.append('  - %d) (%.2f, %.2f, %.2f)' % (
                        i,
                        float(m.get('x', 0.0)),
                        float(m.get('y', 0.0)),
                        float(m.get('z', self.target_z)),
                    ))
                if len(all_events) > max_events_show:
                    lines.append('  - ... (%d more)' % (len(all_events) - max_events_show))
                lines.append('')
            
            # =============== 颜色统计信息（汇总）===============
            lines.append('## Color Targets (Raw Detection Summary)')
            lines.append('')
            raw_targets = report.get('targets_raw', {})
            for color in sorted(raw_targets.keys()):
                items = raw_targets.get(color, [])
                if not items:
                    continue
                lines.append('### %s' % color)
                lines.append('- Count: %d' % len(items))
                for i, item in enumerate(items[:max_raw_show], 1):
                    lines.append('  - %d) Map: (%.2f, %.2f, %.2f), Pixel: (%d, %d), Area: %.0f' % (
                        i,
                        float(item.get('x', 0.0)),
                        float(item.get('y', 0.0)),
                        float(item.get('z', self.target_z)),
                        int(float(item.get('px', 0.0))),
                        int(float(item.get('py', 0.0))),
                        float(item.get('area', 0.0)),
                    ))
                if len(items) > max_raw_show:
                    lines.append('  - ... (%d more)' % (len(items) - max_raw_show))
                lines.append('')
            
            lines.append('---')
            lines.append('**Report generated at:** %s' % time_text)

            with open(self.markdown_report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            rospy.logwarn_throttle(5.0, '写入Markdown报告失败: %s', e)

    def _on_shutdown(self):
        """Stop robot and flush final target report on node shutdown."""
        self._stop_robot()
        self._finalize_patrol_report()

    def _cluster_color_targets(self, color, points):
        if not points:
            return []

        clusters = []
        visited = [False] * len(points)

        for i in range(len(points)):
            if visited[i]:
                continue

            q = deque([i])
            visited[i] = True
            group = []

            while q:
                cur = q.popleft()
                group.append(cur)
                x1 = float(points[cur].get('x', 0.0))
                y1 = float(points[cur].get('y', 0.0))

                for j in range(len(points)):
                    if visited[j]:
                        continue
                    x2 = float(points[j].get('x', 0.0))
                    y2 = float(points[j].get('y', 0.0))
                    # 同簇约束：与当前点近且与当前簇重心也不应过远，防止链式串联误合并
                    if math.hypot(x1 - x2, y1 - y2) <= self.cluster_radius:
                        if group:
                            cxg = sum(float(points[idx].get('x', 0.0)) for idx in group) / float(len(group))
                            cyg = sum(float(points[idx].get('y', 0.0)) for idx in group) / float(len(group))
                            # 严格约束：新点与簇重心距离不超过 cluster_radius + 0.15（从0.35大幅降至0.15）
                            if math.hypot(x2 - cxg, y2 - cyg) > (self.cluster_radius + 0.15):
                                continue
                        visited[j] = True
                        q.append(j)

            group_pts = [points[idx] for idx in group]
            cx = sum(float(p.get('x', 0.0)) for p in group_pts) / float(len(group_pts))
            cy = sum(float(p.get('y', 0.0)) for p in group_pts) / float(len(group_pts))
            cz = sum(float(p.get('z', self.target_z)) for p in group_pts) / float(len(group_pts))
            ts_first = min(float(p.get('ts', 0.0)) for p in group_pts)
            ts_last = max(float(p.get('ts', 0.0)) for p in group_pts)

            clusters.append({
                'color': color,
                'x': float(cx),
                'y': float(cy),
                'z': float(cz),
                'points_count': int(len(group_pts)),
                'first_seen': float(ts_first),
                'last_seen': float(ts_last),
                'members': group_pts,
            })

        return clusters

    def _cluster_all_targets(self):
        all_clusters = []
        for color in sorted(self.detected_targets.keys()):
            pts = self.detected_targets.get(color, [])
            all_clusters.extend(self._cluster_color_targets(color, pts))

        # 第二阶段：同色簇中心再合并，进一步消除单次观测抖动导致的“碎簇”
        all_clusters = self._merge_clusters_by_center(all_clusters)
        # 已知场景只有3个物体，强制再合并到3个
        all_clusters = self._force_cluster_count(all_clusters, target_count=3)

        all_clusters.sort(key=lambda c: c.get('first_seen', 0.0))
        for idx, c in enumerate(all_clusters, 1):
            c['id'] = idx
        return all_clusters

    def _force_cluster_count(self, clusters, target_count=3):
        if not clusters or len(clusters) <= target_count:
            return clusters

        def _merge_two(a, b):
            members = list(a.get('members', [])) + list(b.get('members', []))
            color_counts = {}
            for m in members:
                c = str(m.get('color', 'unknown'))
                color_counts[c] = color_counts.get(c, 0) + 1
            color = max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else str(a.get('color', 'unknown'))
            cx = sum(float(m.get('x', 0.0)) for m in members) / float(max(1, len(members)))
            cy = sum(float(m.get('y', 0.0)) for m in members) / float(max(1, len(members)))
            cz = sum(float(m.get('z', self.target_z)) for m in members) / float(max(1, len(members)))
            ts_first = min(float(m.get('ts', 0.0)) for m in members) if members else 0.0
            ts_last = max(float(m.get('ts', 0.0)) for m in members) if members else 0.0
            return {
                'color': color,
                'x': float(cx),
                'y': float(cy),
                'z': float(cz),
                'points_count': int(len(members)),
                'first_seen': float(ts_first),
                'last_seen': float(ts_last),
                'members': members,
            }

        working = [dict(c) for c in clusters]
        for c in working:
            c['members'] = list(c.get('members', []))

        while len(working) > target_count:
            best_i = -1
            best_j = -1
            best_d = 1e18
            for i in range(len(working)):
                ci = working[i]
                for j in range(i + 1, len(working)):
                    cj = working[j]
                    d = math.hypot(float(ci.get('x', 0.0)) - float(cj.get('x', 0.0)),
                                   float(ci.get('y', 0.0)) - float(cj.get('y', 0.0)))
                    if d < best_d:
                        best_d = d
                        best_i = i
                        best_j = j
            if best_i < 0 or best_j < 0:
                break
            ci = working[best_i]
            cj = working[best_j]
            merged = _merge_two(ci, cj)
            if best_i > best_j:
                best_i, best_j = best_j, best_i
            working.pop(best_j)
            working.pop(best_i)
            working.append(merged)

        return working

    def _merge_clusters_by_center(self, clusters):
        if not clusters:
            return []

        def _make_cluster(color, members):
            cx = sum(float(m.get('x', 0.0)) for m in members) / float(max(1, len(members)))
            cy = sum(float(m.get('y', 0.0)) for m in members) / float(max(1, len(members)))
            cz = sum(float(m.get('z', self.target_z)) for m in members) / float(max(1, len(members)))
            ts_first = min(float(m.get('ts', 0.0)) for m in members) if members else 0.0
            ts_last = max(float(m.get('ts', 0.0)) for m in members) if members else 0.0
            return {
                'color': color,
                'x': float(cx),
                'y': float(cy),
                'z': float(cz),
                'points_count': int(len(members)),
                'first_seen': float(ts_first),
                'last_seen': float(ts_last),
                'members': members,
            }

        working = [dict(c) for c in clusters]
        for c in working:
            c['members'] = list(c.get('members', []))

        changed = True
        while changed:
            changed = False
            best_i = -1
            best_j = -1
            best_d = 1e18

            for i in range(len(working)):
                ci = working[i]
                for j in range(i + 1, len(working)):
                    cj = working[j]
                    if ci.get('color', 'unknown') != cj.get('color', 'unknown'):
                        continue

                    d = math.hypot(float(ci.get('x', 0.0)) - float(cj.get('x', 0.0)),
                                   float(ci.get('y', 0.0)) - float(cj.get('y', 0.0)))
                    if d > self.cluster_merge_radius:
                        continue

                    merged_members = list(ci.get('members', [])) + list(cj.get('members', []))
                    merged_span = self._cluster_span_radius(merged_members)
                    if merged_span > self.cluster_max_span:
                        continue

                    if d < best_d:
                        best_d = d
                        best_i = i
                        best_j = j

            if best_i >= 0 and best_j >= 0:
                ci = working[best_i]
                cj = working[best_j]
                merged = _make_cluster(ci.get('color', 'unknown'), list(ci.get('members', [])) + list(cj.get('members', [])))
                if best_i > best_j:
                    best_i, best_j = best_j, best_i
                working.pop(best_j)
                working.pop(best_i)
                working.append(merged)
                changed = True

        return working

    def _cluster_span_radius(self, members):
        if not members:
            return 0.0
        cx = sum(float(m.get('x', 0.0)) for m in members) / float(len(members))
        cy = sum(float(m.get('y', 0.0)) for m in members) / float(len(members))
        r = 0.0
        for m in members:
            r = max(r, math.hypot(float(m.get('x', 0.0)) - cx, float(m.get('y', 0.0)) - cy))
        return r

    def _print_final_report(self, report=None):
        if self._final_report_printed:
            return
        self._final_report_printed = True

        if report is None:
            report = self._build_target_report()

        all_targets = report.get('targets_clustered', [])
        rospy.loginfo("=== Patrol Report ===")
        rospy.loginfo("Number of targets detected: %d", len(all_targets))
        rospy.loginfo("List of target locations:")
        for idx, t in enumerate(all_targets, 1):
            rospy.loginfo("  %d. color=%s (%.2f, %.2f, %.2f)", idx, t['color'], t['x'], t['y'], t['z'])

    def _capture_home_pose_once(self):
        if self.home_pose is not None:
            return
        if not self.odom_received:
            return

        px, py = self._planning_pose_xy()
        pyaw = self._planning_yaw()
        self.home_pose = (float(px), float(py), float(pyaw))
        rospy.loginfo('已记录起点位姿: (%.2f, %.2f, %.2f)', self.home_pose[0], self.home_pose[1], self.home_pose[2])

    def _update_unknown_history(self):
        if self.map_data is None:
            return
        now = rospy.get_time()
        unknown_cells = int(np.count_nonzero(self.map_data == -1))
        self.unknown_history.append((now, unknown_cells))

        while self.unknown_history and (now - self.unknown_history[0][0]) > self.unknown_stable_window_sec:
            self.unknown_history.popleft()

    def _update_loop_history(self):
        px, py = self._planning_pose_xy()
        if not self.loop_pose_hist:
            self.loop_pose_hist.append((float(px), float(py)))
            self.loop_cumlen_hist.append(0.0)
            self.loop_path_total = 0.0
            return

        lx, ly = self.loop_pose_hist[-1]
        step = math.hypot(float(px) - lx, float(py) - ly)
        if step < self.loop_sample_step:
            return

        self.loop_path_total += step
        self.loop_pose_hist.append((float(px), float(py)))
        self.loop_cumlen_hist.append(float(self.loop_path_total))

    def _bbox_known_ratio(self, min_x, max_x, min_y, max_y):
        if self.map_data is None or self.map_info is None:
            return 0.0
        p0 = self.world_to_grid(min_x, min_y)
        p1 = self.world_to_grid(max_x, max_y)
        if p0 is None or p1 is None:
            return 0.0

        gx0, gy0 = p0
        gx1, gy1 = p1
        x0, x1 = min(gx0, gx1), max(gx0, gx1)
        y0, y1 = min(gy0, gy1), max(gy0, gy1)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        patch = self.map_data[y0:y1 + 1, x0:x1 + 1]
        total = patch.size
        if total <= 0:
            return 0.0
        known = int(np.count_nonzero(patch != -1))
        return float(known) / float(total)

    def _should_start_return_by_loop(self):
        if (not self.enable_loop_return) or (self.map_data is None):
            return False
        now = rospy.get_time()
        if (now - self.start_time) < self.unknown_stable_min_runtime_sec:
            return False
        if self.frontier_missing_since <= 0.0:
            return False
        if (now - self.frontier_missing_since) < self.loop_no_frontier_sec:
            return False

        n = len(self.loop_pose_hist)
        if n < (self.loop_min_points_gap + 2):
            return False

        end_idx = n - 1
        ex, ey = self.loop_pose_hist[end_idx]
        close_idx = -1

        for i in range(0, end_idx - self.loop_min_points_gap):
            ix, iy = self.loop_pose_hist[i]
            if math.hypot(ex - ix, ey - iy) > self.loop_close_dist:
                continue
            path_len = self.loop_cumlen_hist[end_idx] - self.loop_cumlen_hist[i]
            if path_len >= self.loop_min_path_len:
                close_idx = i
                break

        if close_idx < 0:
            return False

        loop_pts = list(self.loop_pose_hist)[close_idx:end_idx + 1]
        xs = [p[0] for p in loop_pts]
        ys = [p[1] for p in loop_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_area = max(0.0, (max_x - min_x) * (max_y - min_y))
        if bbox_area < self.loop_bbox_min_area:
            return False

        known_ratio = self._bbox_known_ratio(min_x, max_x, min_y, max_y)
        if known_ratio < self.loop_known_ratio_thresh:
            return False

        if not self._loop_return_logged:
            rospy.loginfo('闭合回路返航触发: known_ratio=%.2f bbox_area=%.2f', known_ratio, bbox_area)
            self._loop_return_logged = True
        return True

    def _should_start_return(self):
        if self._should_start_return_by_coverage():
            return True
        if self._should_start_return_by_loop():
            return True
        if (rospy.get_time() - self.start_time) < self.unknown_stable_min_runtime_sec:
            return False
        if len(self.unknown_history) < 2:
            return False
        if self.frontier_missing_since <= 0.0:
            return False
        if (rospy.get_time() - self.frontier_missing_since) < self.return_requires_no_frontier_sec:
            return False

        oldest_t, oldest_u = self.unknown_history[0]
        newest_t, newest_u = self.unknown_history[-1]
        if (newest_t - oldest_t) < self.unknown_stable_window_sec * 0.9:
            return False

        reduced = int(oldest_u - newest_u)
        return reduced <= self.unknown_stable_delta_cells

    def _navigate_return_home(self):
        if self.home_pose is None:
            self._capture_home_pose_once()
        if self.home_pose is None:
            return False

        hx, hy, hyaw = self.home_pose
        self.goal_tolerance = self.home_reach_tol
        reached = self.drive_to_target((hx, hy))
        if not reached:
            return False

        yaw_err = self._normalize_angle(hyaw - self._planning_yaw())
        if abs(yaw_err) <= self.home_yaw_tol:
            self._publish_stop('返航到起点并完成朝向对齐')
            return True

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = max(-0.7, min(0.7, 1.6 * yaw_err))
        self._publish_motion(twist)
        return False

    def scan_callback(self, data):
        self.scan_msg = data
        self.laser_ranges = data.ranges

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _sanitize_range(self, r):
        if r is None:
            return float('inf')
        if (not math.isfinite(r)) or r <= 0.0:
            return float('inf')
        return r

    def get_min_range(self, angle_center_rad, angle_width_rad):
        """取某个角度扇区内的最小有效距离（更稳健，避免固定索引导致的越界/分辨率差异）。"""
        if self.scan_msg is None or not self.scan_msg.ranges:
            return float('inf')

        angle_min = self.scan_msg.angle_min
        angle_inc = self.scan_msg.angle_increment
        if angle_inc == 0.0:
            return float('inf')

        center_idx = int(round((angle_center_rad - angle_min) / angle_inc))
        half_span = int(max(0, round((angle_width_rad / 2.0) / angle_inc)))

        start_idx = max(0, center_idx - half_span)
        end_idx = min(len(self.scan_msg.ranges) - 1, center_idx + half_span)

        min_r = float('inf')
        for idx in range(start_idx, end_idx + 1):
            min_r = min(min_r, self._sanitize_range(self.scan_msg.ranges[idx]))
        return min_r

    def odom_callback(self, data):
        self.current_pos.x = data.pose.pose.position.x
        self.current_pos.y = data.pose.pose.position.y
        quat = data.pose.pose.orientation
        quat_list = [quat.x, quat.y, quat.z, quat.w]
        _, _, yaw = euler_from_quaternion(quat_list)
        self.current_yaw = yaw
        self.odom_received = True

    def _update_map_pose_from_tf(self):
        """尝试获取 map->base_* 位姿；失败时回退到/odom位姿。"""
        base_candidates = ("base_footprint", "base_link")
        for base in base_candidates:
            try:
                trans = self.tf_buffer.lookup_transform("map", base, rospy.Time(0), rospy.Duration(0.05))
                self.map_pos_x = float(trans.transform.translation.x)
                self.map_pos_y = float(trans.transform.translation.y)
                q = trans.transform.rotation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                self.map_yaw = float(yaw)
                self.map_pose_valid = True
                return
            except Exception:
                continue
        self.map_pose_valid = False

    def _planning_pose_xy(self):
        if self.map_pose_valid:
            return (self.map_pos_x, self.map_pos_y)
        return (float(self.current_pos.x), float(self.current_pos.y))

    def _planning_yaw(self):
        if self.map_pose_valid:
            return float(self.map_yaw)
        return float(self.current_yaw)

    def _prune_blacklist(self, now):
        while self.frontier_blacklist and (now - self.frontier_blacklist[0][2]) > self.blacklist_ttl_sec:
            self.frontier_blacklist.popleft()

    def _is_blacklisted(self, target_point, now):
        self._prune_blacklist(now)
        tx, ty = float(target_point[0]), float(target_point[1])
        for bx, by, _ in self.frontier_blacklist:
            if math.hypot(tx - bx, ty - by) < self.blacklist_radius:
                return True
        return False

    def _blacklist_target(self, target_point, reason):
        if target_point is None:
            return
        now = rospy.get_time()
        tx, ty = float(target_point[0]), float(target_point[1])
        if self._is_blacklisted((tx, ty), now):
            return
        self.frontier_blacklist.append((tx, ty, now))
        rospy.logwarn("目标加入黑名单(%.2f, %.2f): %s", tx, ty, reason)

    def _blacklist_pose(self, x, y, reason):
        now = rospy.get_time()
        tx, ty = float(x), float(y)
        if self._is_blacklisted((tx, ty), now):
            return
        self.frontier_blacklist.append((tx, ty, now))
        rospy.logwarn("位置加入黑名单(%.2f, %.2f): %s", tx, ty, reason)

    def _stop_robot(self):
        """节点退出时发零速，避免Gazebo保持最后一次速度命令。"""
        stop = Twist()
        try:
            for _ in range(3):
                self.vel_pub.publish(stop)
        except Exception:
            pass

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int16).reshape((self.map_info.height, self.map_info.width))

    def grid_to_world(self, grid_x, grid_y):
        if self.map_info is None:
            return None
        res = self.map_info.resolution
        world_x = self.map_info.origin.position.x + (grid_x + 0.5) * res
        world_y = self.map_info.origin.position.y + (grid_y + 0.5) * res
        return (world_x, world_y)

    def world_to_grid(self, world_x, world_y):
        if self.map_info is None:
            return None
        res = self.map_info.resolution
        gx = int(math.floor((world_x - self.map_info.origin.position.x) / res))
        gy = int(math.floor((world_y - self.map_info.origin.position.y) / res))
        if gx < 0 or gy < 0 or gx >= self.map_info.width or gy >= self.map_info.height:
            return None
        return (gx, gy)

    def _is_free(self, gx, gy):
        return int(self.map_data[gy][gx]) == 0

    def _is_free_mask(self, gx, gy, free_mask):
        return bool(free_mask[gy][gx])

    def _is_unknown(self, gx, gy):
        return int(self.map_data[gy][gx]) == -1

    def _has_unknown_neighbor(self, gx, gy):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if nx < 0 or ny < 0 or nx >= self.map_info.width or ny >= self.map_info.height:
                    continue
                if self._is_unknown(nx, ny):
                    return True
        return False

    def _nearest_free_seed(self, gx, gy, max_radius=10):
        if gx is None or gy is None:
            return None
        if self._is_free(gx, gy):
            return (gx, gy)
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if nx < 0 or ny < 0 or nx >= self.map_info.width or ny >= self.map_info.height:
                        continue
                    if self._is_free(nx, ny):
                        return (nx, ny)
        return None

    def _nearest_free_seed_mask(self, gx, gy, free_mask, max_radius=10):
        if gx is None or gy is None:
            return None
        if self._is_free_mask(gx, gy, free_mask):
            return (gx, gy)
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if nx < 0 or ny < 0 or nx >= self.map_info.width or ny >= self.map_info.height:
                        continue
                    if self._is_free_mask(nx, ny, free_mask):
                        return (nx, ny)
        return None

    def _ensure_inflate_offsets(self):
        if self.map_info is None or self.map_info.resolution <= 0.0:
            return
        radius_m = max(self.robot_radius_m, self.inflation_radius_m)
        radius_cells = int(math.ceil(radius_m / self.map_info.resolution))
        if radius_cells == self._inflate_radius_cells and self._inflate_offsets:
            return

        self._inflate_radius_cells = radius_cells
        offsets = []
        rr2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx + dy * dy) <= rr2:
                    offsets.append((dx, dy))
        self._inflate_offsets = offsets

    def _build_free_mask(self):
        """构建膨胀后可通行栅格：只允许足够远离障碍的free单元通过。"""
        if self.map_data is None or self.map_info is None:
            return None

        self._ensure_inflate_offsets()

        free_mask = (self.map_data == 0)
        occ = np.argwhere(self.map_data > self.occ_threshold)
        if occ.size == 0 or not self._inflate_offsets:
            return free_mask

        h, w = free_mask.shape
        for oy, ox in occ:
            for dx, dy in self._inflate_offsets:
                nx = int(ox + dx)
                ny = int(oy + dy)
                if 0 <= nx < w and 0 <= ny < h:
                    free_mask[ny, nx] = False
        return free_mask

    def _is_recent_frontier(self, world_point):
        if world_point is None:
            return False
        tx, ty = float(world_point[0]), float(world_point[1])
        for fx, fy in self.recent_frontiers:
            if math.hypot(tx - fx, ty - fy) < self.revisit_radius:
                return True
        return False

    def _frontier_unknown_gain(self, gx, gy):
        if self.map_data is None:
            return 0
        w = max(1, int(self.frontier_unknown_window_cells))
        x0 = max(0, int(gx) - w)
        x1 = min(int(self.map_info.width) - 1, int(gx) + w)
        y0 = max(0, int(gy) - w)
        y1 = min(int(self.map_info.height) - 1, int(gy) + w)
        patch = self.map_data[y0:y1 + 1, x0:x1 + 1]
        return int(np.count_nonzero(patch == -1))

    def _reachable_frontiers(self):
        """BFS遍历可达free区域，返回frontier集合、父节点树、距离表、seed。"""
        if self.map_data is None or self.map_info is None:
            return None, None, None, None

        free_mask = self._build_free_mask()
        if free_mask is None:
            return None, None, None, None

        px, py = self._planning_pose_xy()
        start = self.world_to_grid(px, py)
        if start is None:
            return None, None, None, None

        seed = self._nearest_free_seed_mask(start[0], start[1], free_mask, max_radius=20)
        if seed is None:
            return None, None, None, None

        q = deque([seed])
        parent = {seed: None}
        dist = {seed: 0}
        frontiers = set()

        while q:
            gx, gy = q.popleft()
            if self._has_unknown_neighbor(gx, gy):
                frontiers.add((gx, gy))

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = gx + dx, gy + dy
                if nx < 0 or ny < 0 or nx >= self.map_info.width or ny >= self.map_info.height:
                    continue
                if (nx, ny) in parent:
                    continue
                if self._is_free_mask(nx, ny, free_mask):
                    parent[(nx, ny)] = (gx, gy)
                    dist[(nx, ny)] = dist[(gx, gy)] + 1
                    q.append((nx, ny))

        return frontiers, parent, dist, seed

    def _cluster_frontiers(self, frontier_cells):
        if not frontier_cells:
            return []
        remaining = set(frontier_cells)
        clusters = []
        while remaining:
            root = remaining.pop()
            q = deque([root])
            cluster = [root]
            while q:
                cx, cy = q.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb = (cx + dx, cy + dy)
                        if nb in remaining:
                            remaining.remove(nb)
                            q.append(nb)
                            cluster.append(nb)
            clusters.append(cluster)
        return clusters

    def _path_from_parent(self, parent, goal_cell):
        path = []
        cur = goal_cell
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path

    def _path_to_waypoints(self, grid_path):
        if not grid_path:
            return []
        waypoints = []
        for i in range(0, len(grid_path), self.waypoint_step):
            wp = self.grid_to_world(grid_path[i][0], grid_path[i][1])
            if wp is not None:
                waypoints.append(wp)
        end_wp = self.grid_to_world(grid_path[-1][0], grid_path[-1][1])
        if end_wp is not None:
            if not waypoints:
                waypoints.append(end_wp)
            else:
                last = waypoints[-1]
                if math.hypot(last[0] - end_wp[0], last[1] - end_wp[1]) > 1e-3:
                    waypoints.append(end_wp)
        return waypoints

    def _plan_frontier_path(self):
        """选择最优可达frontier簇并返回(frontier_world, waypoints)。"""
        if self.map_data is None or self.map_info is None:
            return None, None

        frontiers, parent, dist, _ = self._reachable_frontiers()
        if frontiers is None:
            return None, None
        if not frontiers:
            self.frontiers_exhausted = True
            return None, None
        self.frontiers_exhausted = False

        clusters = self._cluster_frontiers(frontiers)
        if not clusters:
            self.frontiers_exhausted = True
            return None, None
        self.frontiers_exhausted = False

        usable_clusters = [c for c in clusters if len(c) >= self.frontier_min_cluster]
        if not usable_clusters:
            usable_clusters = clusters

        now = rospy.get_time()
        candidates = []
        px, py = self._planning_pose_xy()

        for cluster in usable_clusters:
            rep = max(cluster, key=lambda c: (self._frontier_unknown_gain(c[0], c[1]), -dist.get(c, 10**9)))
            d = float(dist.get(rep, 10**9))
            rep_world = self.grid_to_world(rep[0], rep[1])
            if rep_world is None:
                continue
            gain = self._frontier_unknown_gain(rep[0], rep[1])
            # 跳过离机器人过近的frontier，避免反复选中“脚边目标”导致原地重规划
            if math.hypot(rep_world[0] - px, rep_world[1] - py) < self.min_frontier_goal_dist:
                continue
            # 未知增益太小且簇也不大时跳过，减少在已建图区域来回震荡
            if gain < self.frontier_min_unknown_gain and len(cluster) < max(self.frontier_min_cluster * 2, 12):
                continue
            # 黑名单目标直接禁选，避免反复选回同一区域
            if self._is_blacklisted(rep_world, now):
                continue
            # 优先选择未知增益更高的frontier，再考虑簇大小与距离；最近访问区域强降权
            revisit_penalty = self.frontier_revisit_penalty if self._is_recent_frontier(rep_world) else 0.0
            score = (self.frontier_size_weight * len(cluster)) + (self.frontier_gain_weight * gain) - (self.frontier_dist_weight * d) - revisit_penalty
            candidates.append((score, rep, rep_world, gain, d, len(cluster)))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0], reverse=True)

        # 第一轮：严格跳过最近区域；第二轮：放宽，避免无目标可选
        for strict_recent in (True, False):
            for score, rep, rep_world, gain, d, csz in candidates:
                if strict_recent and self._is_recent_frontier(rep_world):
                    continue

                path_grid = self._path_from_parent(parent, rep)
                if len(path_grid) < self.min_path_cells and len(candidates) > 1:
                    continue

                waypoints = self._path_to_waypoints(path_grid)
                if len(waypoints) < 2 and len(candidates) > 1:
                    continue

                rospy.loginfo_throttle(1.0,
                                       "frontier评分选中: score=%.1f gain=%d cluster=%d dist=%.1f",
                                       score, int(gain), int(csz), d)

                return rep_world, waypoints

        return None, None

    def _check_and_handle_stuck(self):
        now = rospy.get_time()
        px, py = self._planning_pose_xy()
        self._pose_hist.append((now, px, py))
        if len(self._pose_hist) < self._pose_hist.maxlen:
            return False

        t0, x0, y0 = self._pose_hist[0]
        t1, x1, y1 = self._pose_hist[-1]
        if (t1 - t0) < self._stuck_time_sec:
            return False

        moved = math.hypot(x1 - x0, y1 - y0)
        if moved < self._stuck_dist_eps and now >= self._recovery_until:
            # 进入短暂恢复：原地旋转，改变turn_dir避免重复
            self.turn_dir *= -1.0
            self._recovery_until = now + 2.0
            self._stuck_event_count += 1
            rospy.logwarn("检测到疑似卡死(%.3fm/%.1fs)，进入恢复旋转", moved, (t1 - t0))
            return True
        return False

    def _update_progress(self):
        now = rospy.get_time()
        px, py = self._planning_pose_xy()
        if self._last_progress_pose is None:
            self._last_progress_pose = (px, py)
            self._last_progress_time = now
            return
        moved = math.hypot(px - self._last_progress_pose[0], py - self._last_progress_pose[1])
        if moved > self.progress_eps:
            self._last_progress_pose = (px, py)
            self._last_progress_time = now

    def _no_progress(self):
        if self._last_progress_time <= 0.0:
            return False
        return (rospy.get_time() - self._last_progress_time) > self.no_progress_timeout

    def _publish_recovery(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.turn_dir * 0.7
        self._publish_motion(twist)

    def _publish_escape_motion(self):
        """强制脱困动作：短暂后退并转向，脱离局部循环。"""
        twist = Twist()
        # 先后退再前探，配合较大角速度离开局部势阱
        phase = int(rospy.get_time() * 2.0) % 4
        if phase in (0, 1):
            twist.linear.x = -0.04
        else:
            twist.linear.x = 0.04
        twist.angular.z = self.turn_dir * 0.6
        self._publish_motion(twist)

    def _start_obstacle_escape(self, turn_dir, now=None):
        if now is None:
            now = rospy.get_time()
        self.obstacle_escape_dir = 1.0 if turn_dir >= 0.0 else -1.0
        self.obstacle_escape_until = now + self.obstacle_escape_hold_sec
        self.obstacle_clear_streak = 0

    def _clear_obstacle_escape(self):
        self.obstacle_escape_dir = 0.0
        self.obstacle_escape_until = 0.0
        self.obstacle_clear_streak = 0

    def explore_without_map(self):
        """无地图时的保底探索 - 简化版本，使用动态窗口法则。"""
        front = self.get_min_range(0.0, math.radians(25.0))
        front_left = self.get_min_range(math.radians(35.0), math.radians(40.0))
        front_right = self.get_min_range(-math.radians(35.0), math.radians(40.0))
        left = self.get_min_range(math.radians(90.0), math.radians(30.0))
        right = self.get_min_range(-math.radians(90.0), math.radians(30.0))

        twist = Twist()
        
        # 危险检查
        if front < self.danger_distance:
            twist.linear.x = 0.0
            twist.angular.z = 0.8 if front_left > front_right else -0.8
        # 安全距离内减速
        elif front < self.safe_distance:
            twist.linear.x = 0.06
            left_score = min(front_left, left)
            right_score = min(front_right, right)
            if left_score > right_score + 0.2:
                twist.angular.z = 0.4
            elif right_score > left_score + 0.2:
                twist.angular.z = -0.4
            else:
                if not math.isfinite(left) or left < self.wall_dist:
                    twist.angular.z = -0.3
                else:
                    twist.angular.z = 0.3
        # 前方开阔，自由探索
        else:
            twist.linear.x = 0.15
            left_score = min(front_left, left)
            right_score = min(front_right, right)
            if left_score > right_score:
                twist.angular.z = -0.2
            else:
                twist.angular.z = 0.2

        self._publish_motion(twist)
    def _goal_bearing_and_dist(self, target_point):
        tx, ty = float(target_point[0]), float(target_point[1])
        px, py = self._planning_pose_xy()
        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw = self._planning_yaw()
        bearing = self._normalize_angle(desired_yaw - yaw)
        return bearing, dist

    def _is_goal_direction_clear(self, bearing, dist):
        # 用指向目标方向的扇区，判断是否“近似直线可达”
        r = self.get_min_range(bearing, math.radians(12.0))
        if not math.isfinite(r):
            return True
        return r > max(0.0, dist - 0.15)

    def navigate_bug_to_target(self, target_point):
        """
        改进的避障导航 - 使用动态窗口法则代替复杂的Bug算法。
        核心策略：
        1. 优先安全（避开障碍）
        2. 其次靠近目标
        3. 最后考虑效率
        """
        if target_point is None:
            return False

        bearing, dist = self._goal_bearing_and_dist(target_point)
        
        # 到达目标判定
        if dist < self.goal_tolerance:
            self._publish_stop('到达当前路点，等待下一个目标')
            return True

        # ===== 获取关键方向的距离信息 =====
        front = self.get_min_range(0.0, math.radians(30.0))          # 正前方
        front_left = self.get_min_range(math.radians(45.0), math.radians(40.0))   # 左前
        front_right = self.get_min_range(-math.radians(45.0), math.radians(40.0))  # 右前
        left = self.get_min_range(math.radians(90.0), math.radians(30.0))         # 左侧
        right = self.get_min_range(-math.radians(90.0), math.radians(30.0))       # 右侧
        now = rospy.get_time()

        twist = Twist()
        left_safety = min(front_left, left)
        right_safety = min(front_right, right)
        side_safety = min(front_left, front_right, left, right)

        # 逃逸锁：一旦开始绕障，必须先持续离开障碍边界，再恢复按目标方向修正。
        # 这可以避免“右转后下一帧立刻左转”导致的擦碰。
        if self.obstacle_escape_dir != 0.0:
            clear_now = (front > self.escape_min_forward_clear) and (side_safety > self.escape_min_side_clear)
            if clear_now:
                self.obstacle_clear_streak += 1
            else:
                self.obstacle_clear_streak = 0

            twist.linear.x = 0.04
            twist.angular.z = 0.40 * self.obstacle_escape_dir
            self._log_nav_status('avoid_escape_lock',
                                 '开始执行避障动作: 逃逸保持中 dir=%s front=%.2f side=%.2f' %
                                 ('left' if self.obstacle_escape_dir > 0.0 else 'right', front, side_safety))

            # 至少保持一小段逃逸时间，并且连续若干帧确认已脱离安全边界后再释放。
            if clear_now and now >= self.obstacle_escape_until and self.obstacle_clear_streak >= self.obstacle_clear_needed:
                self._clear_obstacle_escape()
            self._publish_motion(twist)
            return False
        
        # ===== 第一层：临界安全检查 =====
        if front < self.critical_distance:
            # 极其接近，立即紧急后退
            twist.linear.x = -0.10
            turn_dir = 1.0 if front_left >= front_right else -1.0
            twist.angular.z = 1.2 * turn_dir
            self._start_obstacle_escape(turn_dir, now)
            self._log_nav_status('avoid_critical',
                                 '开始执行避障动作: 紧急后退+转向 front=%.2f dir=%s' %
                                 (front, 'left' if turn_dir > 0.0 else 'right'))
            self._publish_motion(twist)
            rospy.logwarn("紧急后退: 前方距离过近 %.2fm", front)
            return False
        
        # ===== 第二层：危险距离检查 =====
        if front < self.danger_distance:
            # 危险距离，停止前进并转向最安全方向
            twist.linear.x = 0.0
            # 转向离子最远的方向
            turn_dir = 1.0 if left_safety >= right_safety else -1.0
            twist.angular.z = 0.8 * turn_dir
            self._log_nav_status('avoid_danger',
                                 '开始执行避障动作: 原地转向避障 front=%.2f dir=%s' %
                                 (front, 'left' if turn_dir > 0.0 else 'right'))
            self._publish_motion(twist)
            rospy.loginfo("避障转向: 前方距离 %.2fm", front)
            return False
        
        # ===== 第三层：安全距离检查 =====
        if front < self.safe_distance:
            # 进入安全边界，需要减速或转向
            twist.linear.x = 0.05  # 非常低速试探
            
            # 评估各个方向的安全性
            left_score = min(front_left, left)
            right_score = min(front_right, right)
            
            # 如果一侧明显更安全，则转向
            if left_score > right_score + 0.15:
                twist.angular.z = 0.5  # 温和左转
            elif right_score > left_score + 0.15:
                twist.angular.z = -0.5  # 温和右转
            else:
                # 两侧差不多，根据目标方向微调
                turn_dir = 1.0 if (bearing >= 0.0) else -1.0
                twist.angular.z = 0.3 * bearing if abs(bearing) < math.pi/2 else 0.4 * turn_dir            
            self._publish_motion(twist)
            self._log_nav_status('avoid_safe', '开始执行避障动作: 安全边界减速绕行 front=%.2f side=%.2f' % (front, side_safety))
            return False

        # 只要侧前方或侧面已经逼近膨胀区，就提前降速并纠偏，避免擦到已膨胀区域边界。
        if side_safety < self.safe_distance:
            twist.linear.x = 0.04
            if left_safety > right_safety + 0.10:
                twist.angular.z = 0.35
                self._start_obstacle_escape(1.0, now)
            elif right_safety > left_safety + 0.10:
                twist.angular.z = -0.35
                self._start_obstacle_escape(-1.0, now)
            else:
                turn_dir = 1.0 if (bearing >= 0.0) else -1.0
                twist.angular.z = 0.25 * bearing if abs(bearing) < math.pi / 2 else 0.35 * turn_dir
                self._start_obstacle_escape(turn_dir, now)
            self._publish_motion(twist)
            self._log_nav_status('avoid_side', '开始执行避障动作: 侧向避障纠偏 side=%.2f' % side_safety)
            return False
        
        # ===== 第四层：正常导航 =====
        # 前方相对安全，可以靠近目标
        
        # 线性速度：基于前方距离自适应调整
        if front > 1.0:
            # 前方很开阔，可以快速前进
            twist.linear.x = min(self.max_linear, 0.28)
        elif front > 0.75:
            # 前方开阔，中速前进
            twist.linear.x = min(self.max_linear, 0.20)
        else:
            # 前方接近安全距离，低速前进
            twist.linear.x = min(self.max_linear, 0.12)
        
        # 角速度：靠近目标方向
        if abs(bearing) > self.yaw_tolerance:
            # 偏离目标较远，优先转向
            twist.linear.x *= 0.6  # 转向时减速
            k_ang = 1.2
            twist.angular.z = max(-self.max_angular, min(self.max_angular, k_ang * bearing))
        else:
            # 已经对准目标方向，直行
            twist.angular.z = 0.0

        # 当侧向接近障碍时，降低前进速度并向更安全的一侧偏转，避免穿过膨胀边界。
        if side_safety < 1.0:
            twist.linear.x = min(twist.linear.x, 0.16)
            if left_safety > right_safety + 0.10:
                twist.angular.z = max(twist.angular.z, 0.12)
            elif right_safety > left_safety + 0.10:
                twist.angular.z = min(twist.angular.z, -0.12)
        elif side_safety < 1.4:
            twist.linear.x = min(twist.linear.x, 0.22)
        
        self._publish_motion(twist)
        self._log_nav_status('nav_tracking', '导航中: 跟踪路点 dist=%.2f front=%.2f side=%.2f' % (dist, front, side_safety))
        return False

    def drive_to_target(self, target_point):
        return self.navigate_bug_to_target(target_point)

    def run(self):
        rospy.loginfo("=== Patrol task started ===")
        while not rospy.is_shutdown():
            self._update_map_pose_from_tf()
            self._update_progress()
            self._update_loop_history()
            self._capture_home_pose_once()

            if self.scan_msg is None or len(self.laser_ranges) == 0:
                self._publish_stop('等待激光雷达数据')
                rospy.logwarn_throttle(2.0, "等待激光雷达数据...")
                self.rate.sleep()
                continue

            if not self.odom_received:
                self._publish_stop('等待里程计数据')
                rospy.logwarn_throttle(2.0, "等待/odom数据...")
                self.rate.sleep()
                continue

            now = rospy.get_time()
            if now < self._recovery_until:
                self._publish_recovery()
                self.rate.sleep()
                continue

            if now < self.force_escape_until:
                self._publish_escape_motion()
                self.rate.sleep()
                continue

            if now < self.local_minima_until:
                self._publish_escape_motion()
                self.rate.sleep()
                continue

            if self.map_data is None or self.map_info is None:
                rospy.logwarn_throttle(2.0, "等待/map数据...（地图未就绪时不会判定探索完成）")
                self.explore_without_map()
                self._check_and_handle_stuck()
                self.rate.sleep()
                continue

            self._update_unknown_history()

            if (not self.return_mode) and ((now - self.start_time) >= self.force_return_time_sec):
                self.return_mode = True
                self.path_waypoints = []
                self.current_frontier = None
                self.wp_index = 0
                if not self._force_return_logged:
                    rospy.loginfo('已基本探测完所有未知区域 开始返回出发点。')
                    self._force_return_logged = True

            if (not self.return_mode) and self._should_start_return():
                self.return_mode = True
                self.path_waypoints = []
                self.current_frontier = None
                self.wp_index = 0
                rospy.loginfo('未知区域在较长时间内几乎不再变化，开始返航到起点...')

            if self.return_mode and (not self.return_completed):
                done = self._navigate_return_home()
                if done:
                    self.return_completed = True
                    self._publish_stop('返航完成，任务结束')
                    rospy.loginfo('巡逻任务圆满完成')
                    self._finalize_patrol_report()
                    break
                self.rate.sleep()
                continue

            has_unknown = bool(np.any(self.map_data == -1))
            if has_unknown:
                self.no_unknown_count = 0
            else:
                self.no_unknown_count += 1

            if self.no_unknown_count >= self.no_unknown_need:
                if not self.return_mode:
                    self.return_mode = True
                    self.path_waypoints = []
                    self.current_frontier = None
                    self.wp_index = 0
                    rospy.loginfo("地图未知区域清零，切换返航模式...")

            need_replan = (not self.path_waypoints) or (self.wp_index >= len(self.path_waypoints))
            # 仅在无路径/无进展时重规划，避免目标频繁跳变造成局部振荡
            need_replan = need_replan or (((now - self.last_plan_time) > self.replan_interval) and self.current_frontier is None)
            need_replan = need_replan or self._no_progress()

            if self._no_progress() and self.current_frontier is not None and (now - self.last_forced_switch_time) > self.forced_switch_cooldown:
                self._blacklist_target(self.current_frontier, "长时间无进展，强制换目标")
                self.last_forced_switch_time = now
                self.current_frontier = None
                self.path_waypoints = []
                self.wp_index = 0
                self.force_escape_until = now + 2.0
                self._last_progress_pose = self._planning_pose_xy()
                self._last_progress_time = now
                # 不在这里清零stuck计数，让连续卡死策略保持敏感

            if need_replan:
                frontier_world, waypoints = self._plan_frontier_path()
                self.last_plan_time = now

                if frontier_world is not None and waypoints:
                    self.frontiers_exhausted = False
                    self.frontier_missing_since = 0.0
                    old_frontier = self.current_frontier
                    self.current_frontier = frontier_world
                    self.recent_frontiers.append((frontier_world[0], frontier_world[1]))
                    self.path_waypoints = waypoints
                    self.wp_index = 0
                    # 只有目标明显改变时才重置卡死计数，避免同一目标无限重试
                    if old_frontier is None or math.hypot(old_frontier[0] - frontier_world[0], old_frontier[1] - frontier_world[1]) > 0.35:
                        self._stuck_event_count = 0
                    rospy.loginfo_throttle(1.0, "选择frontier目标：(%.2f, %.2f), 路径点:%d",
                                           frontier_world[0], frontier_world[1], len(waypoints))
                else:
                    if has_unknown:
                        if self.frontier_missing_since <= 0.0:
                            self.frontier_missing_since = now
                        if self.frontiers_exhausted:
                            no_frontier_sec = now - self.frontier_missing_since
                            runtime_sec = now - self.start_time
                            if runtime_sec >= self.unknown_stable_min_runtime_sec and no_frontier_sec >= self.return_requires_no_frontier_sec:
                                self.return_mode = True
                                self.path_waypoints = []
                                self.current_frontier = None
                                self.wp_index = 0
                                rospy.loginfo('frontier已耗尽，满足返航条件，开始返航到起点...')
                                self.rate.sleep()
                                continue
                        if self._should_start_return_by_coverage():
                            self.return_mode = True
                            self.path_waypoints = []
                            self.current_frontier = None
                            self.wp_index = 0
                            rospy.loginfo('未知区域已基本完成，开始返航到起点...')
                            self.rate.sleep()
                            continue
                        # 没有可规划frontier时持续探索，避免提前停机
                        self.explore_without_map()
                        stuck_event = self._check_and_handle_stuck()
                        if stuck_event and self.current_frontier is not None and self._stuck_event_count >= self.max_stuck_before_replan:
                            self._blacklist_target(self.current_frontier, "无可规划路径且连续卡死")
                            self.current_frontier = None
                            self.path_waypoints = []
                            self.wp_index = 0
                            self._stuck_event_count = 0
                        self.rate.sleep()
                        continue

            if not self.path_waypoints or self.wp_index >= len(self.path_waypoints):
                if has_unknown and self.frontier_missing_since <= 0.0:
                    self.frontier_missing_since = now
                self._log_nav_status('explore_nominal', '当前无可用路径，执行保底探索并等待新frontier')
                self.explore_without_map()
                self._check_and_handle_stuck()
                self.rate.sleep()
                continue

            target_wp = self.path_waypoints[self.wp_index]
            reached = self.drive_to_target(target_wp)
            if reached:
                self.wp_index += 1
                self._stuck_event_count = 0
                if self.wp_index >= len(self.path_waypoints):
                    self.current_frontier = None
                    self.path_waypoints = []
                    self.wp_index = 0
                    self._stuck_event_count = 0

            stuck_event = self._check_and_handle_stuck()
            if stuck_event and self.current_frontier is not None and self._stuck_event_count >= self.max_stuck_before_replan:
                self._blacklist_target(self.current_frontier, "连续卡死，触发重规划")
                self.current_frontier = None
                self.path_waypoints = []
                self.wp_index = 0
                self._stuck_event_count = 0
                self.local_minima_count += 1
                # 显式进入脱困模式，先脱离局部最优再重新选frontier
                self.local_minima_until = now + min(6.0, 2.0 + 0.8 * self.local_minima_count)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        solver = MazeSolver()
        solver.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序中断")
    except Exception as e:
        rospy.logerr("运行出错: %s", e)
