#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import math
from collections import deque
import tf2_ros
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


class MazeSolver:
    def __init__(self):
        rospy.init_node('maze_solver', anonymous=True)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # 机器人位姿与激光
        self.current_pos = Point()
        self.current_yaw = 0.0
        self.odom_received = False
        self.scan_msg = None
        self.laser_ranges = []
        self.turn_dir = 1.0
        self.rate = rospy.Rate(10)

        # 控制参数
        self.safe_distance = 0.45
        self.max_linear = 0.30
        self.max_angular = 1.2
        self.goal_tolerance = 0.28
        self.yaw_tolerance = 0.35

        # 保底墙跟随
        self.wall_dist = 0.45
        self.wall_kp = 1.5
        self.wall_linear = 0.18

        # Bug导航状态
        self.nav_state = "GOTO"
        self._wall_hit_dist = None
        self._wall_follow_enter_time = 0.0

        # 路径与frontier状态
        self.current_frontier = None
        self.path_waypoints = []
        self.wp_index = 0
        self.waypoint_step = 3
        self.waypoint_reach_tol = 0.22
        self.replan_interval = 2.0
        self.last_plan_time = 0.0
        self.min_path_cells = 10

        # frontier选取参数
        self.frontier_min_cluster = 6
        self.frontier_blacklist = deque(maxlen=100)  # (x, y, ts)
        self.blacklist_radius = 0.5
        self.blacklist_ttl_sec = 60.0
        self.recent_frontiers = deque(maxlen=20)  # (x, y)
        self.revisit_radius = 1.0
        self.last_forced_switch_time = 0.0
        self.forced_switch_cooldown = 3.0
        self.force_escape_until = 0.0

        # 地图膨胀参数：把靠墙太近的free栅格视为不可通行，减少“理论可达/实际过不去”
        self.robot_radius_m = 0.18
        self.clearance_m = 0.08
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

        rospy.on_shutdown(self._stop_robot)

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

    def _stop_robot(self):
        """节点退出时发零速，避免Gazebo保持最后一次速度命令。"""
        stop = Twist()
        try:
            for _ in range(3):
                self.vel_pub.publish(stop)
        except Exception:
            pass

    def avoid_obstacle(self):
        twist = Twist()
        front_range = self.get_min_range(0.0, math.radians(30.0))
        left_range = self.get_min_range(math.pi / 2.0, math.radians(30.0))
        right_range = self.get_min_range(-math.pi / 2.0, math.radians(30.0))

        if front_range < self.safe_distance:
            twist.linear.x = 0.0
            twist.angular.z = self.turn_dir * 0.9
            if left_range < right_range:
                self.turn_dir = 1.0
            else:
                self.turn_dir = -1.0
        else:
            twist.linear.x = 0.24
            twist.angular.z = 0.0
        self.vel_pub.publish(twist)

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
        radius_cells = int(math.ceil((self.robot_radius_m + self.clearance_m) / self.map_info.resolution))
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
        occ = np.argwhere(self.map_data > 50)
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
            return None, None

        clusters = self._cluster_frontiers(frontiers)
        if not clusters:
            return None, None

        usable_clusters = [c for c in clusters if len(c) >= self.frontier_min_cluster]
        if not usable_clusters:
            usable_clusters = clusters

        best = None
        best_score = -1e18
        now = rospy.get_time()
        candidates = []

        for cluster in usable_clusters:
            rep = min(cluster, key=lambda c: dist.get(c, 10**9))
            d = float(dist.get(rep, 10**9))
            rep_world = self.grid_to_world(rep[0], rep[1])
            if rep_world is None:
                continue
            # 黑名单目标直接禁选，避免反复选回同一区域
            if self._is_blacklisted(rep_world, now):
                continue
            # 大簇优先、距离次优先；最近访问过的区域降权
            revisit_penalty = 8.0 if self._is_recent_frontier(rep_world) else 0.0
            score = (2.5 * len(cluster)) - (0.18 * d) - revisit_penalty
            candidates.append((score, rep, rep_world))

            if score > best_score:
                best_score = score
                best = rep

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0], reverse=True)

        # 第一轮：严格跳过最近区域；第二轮：放宽，避免无目标可选
        for strict_recent in (True, False):
            for _, rep, rep_world in candidates:
                if strict_recent and self._is_recent_frontier(rep_world):
                    continue

                path_grid = self._path_from_parent(parent, rep)
                if len(path_grid) < self.min_path_cells and len(candidates) > 1:
                    continue

                waypoints = self._path_to_waypoints(path_grid)
                if len(waypoints) < 2 and len(candidates) > 1:
                    continue

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
        self.vel_pub.publish(twist)

    def _publish_escape_motion(self):
        """强制脱困动作：短暂后退并转向，脱离局部循环。"""
        twist = Twist()
        # 先后退再前探，配合较大角速度离开局部势阱
        phase = int(rospy.get_time() * 2.0) % 4
        if phase in (0, 1):
            twist.linear.x = -0.06
        else:
            twist.linear.x = 0.05
        twist.angular.z = self.turn_dir * 1.0
        self.vel_pub.publish(twist)

    def explore_without_map(self):
        """无/map时的保底探索：右手沿墙 + 遇障转向。"""
        # 右侧与前方距离
        front = self.get_min_range(0.0, math.radians(25.0))
        right = self.get_min_range(-math.pi / 2.0, math.radians(35.0))
        front_right = self.get_min_range(-math.pi / 4.0, math.radians(35.0))

        twist = Twist()
        if front < self.safe_distance:
            twist.linear.x = 0.0
            twist.angular.z = 0.8
        else:
            twist.linear.x = self.wall_linear
            # 没墙时倾向右转去“找墙”
            if not math.isfinite(right) or right == float('inf'):
                twist.angular.z = -0.4
            else:
                err = self.wall_dist - right
                twist.angular.z = max(-self.max_angular, min(self.max_angular, -self.wall_kp * err))
            # 如果右前太近，稍微左拐
            if front_right < (self.wall_dist * 0.9):
                twist.angular.z = max(twist.angular.z, 0.4)

        self.vel_pub.publish(twist)

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
        if target_point is None:
            return False

        bearing, dist = self._goal_bearing_and_dist(target_point)
        if dist < self.goal_tolerance:
            stop = Twist()
            self.vel_pub.publish(stop)
            self.nav_state = "GOTO"
            self._wall_hit_dist = None
            return True

        front = self.get_min_range(0.0, math.radians(25.0))

        now = rospy.get_time()
        if self.nav_state == "GOTO":
            if front < self.safe_distance:
                self.nav_state = "WALL"
                self._wall_hit_dist = dist
                self._wall_follow_enter_time = now
            else:
                twist = Twist()
                k_ang = 1.4
                twist.angular.z = max(-self.max_angular, min(self.max_angular, k_ang * bearing))
                if abs(bearing) > self.yaw_tolerance:
                    twist.linear.x = 0.0
                else:
                    # 前方更空旷时加速，靠近障碍时自动降速
                    if front > 1.0:
                        twist.linear.x = min(self.max_linear, 0.30)
                    elif front > 0.7:
                        twist.linear.x = min(self.max_linear, 0.24)
                    else:
                        twist.linear.x = min(self.max_linear, 0.16)
                self.vel_pub.publish(twist)
                return False

        right = self.get_min_range(-math.pi / 2.0, math.radians(35.0))
        front_right = self.get_min_range(-math.pi / 4.0, math.radians(35.0))

        twist = Twist()
        if front < self.safe_distance:
            twist.linear.x = 0.0
            twist.angular.z = 1.1
        else:
            twist.linear.x = self.wall_linear
            if not math.isfinite(right) or right == float('inf'):
                twist.angular.z = -0.6
            else:
                err = self.wall_dist - right
                twist.angular.z = max(-self.max_angular, min(self.max_angular, -self.wall_kp * err))
            if front_right < (self.wall_dist * 0.9):
                twist.angular.z = max(twist.angular.z, 0.8)

        self.vel_pub.publish(twist)

        if self._wall_hit_dist is not None:
            clear = self._is_goal_direction_clear(bearing, dist)
            improving = dist < (self._wall_hit_dist - 0.10)
            if (now - self._wall_follow_enter_time) > 1.0 and clear and improving and front > (self.safe_distance + 0.05):
                self.nav_state = "GOTO"
                self._wall_hit_dist = None

        return False

    def drive_to_target(self, target_point):
        return self.navigate_bug_to_target(target_point)

    def run(self):
        rospy.loginfo("迷宫探索启动！")
        while not rospy.is_shutdown():
            self._update_map_pose_from_tf()
            self._update_progress()

            if self.scan_msg is None or len(self.laser_ranges) == 0:
                rospy.logwarn_throttle(2.0, "等待激光雷达数据...")
                self.rate.sleep()
                continue

            if not self.odom_received:
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

            has_unknown = bool(np.any(self.map_data == -1))
            if has_unknown:
                self.no_unknown_count = 0
            else:
                self.no_unknown_count += 1

            if self.no_unknown_count >= self.no_unknown_need:
                self.vel_pub.publish(Twist())
                rospy.loginfo("探索完成，小车已停止")
                break

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
                self.nav_state = "GOTO"
                self._wall_hit_dist = None
                self.force_escape_until = now + 2.0
                self._last_progress_pose = self._planning_pose_xy()
                self._last_progress_time = now
                # 不在这里清零stuck计数，让连续卡死策略保持敏感

            if need_replan:
                frontier_world, waypoints = self._plan_frontier_path()
                self.last_plan_time = now

                if frontier_world is not None and waypoints:
                    old_frontier = self.current_frontier
                    self.current_frontier = frontier_world
                    self.recent_frontiers.append((frontier_world[0], frontier_world[1]))
                    self.path_waypoints = waypoints
                    self.wp_index = 0
                    self.nav_state = "GOTO"
                    # 只有目标明显改变时才重置卡死计数，避免同一目标无限重试
                    if old_frontier is None or math.hypot(old_frontier[0] - frontier_world[0], old_frontier[1] - frontier_world[1]) > 0.35:
                        self._stuck_event_count = 0
                    rospy.loginfo_throttle(1.0, "选择frontier目标：(%.2f, %.2f), 路径点:%d",
                                           frontier_world[0], frontier_world[1], len(waypoints))
                else:
                    if has_unknown:
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
                self.nav_state = "GOTO"
                self._wall_hit_dist = None
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
