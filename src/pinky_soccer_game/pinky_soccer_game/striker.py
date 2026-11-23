#!/usr/bin/env python3
"""Striker: Nav2 for global movement + direct control for the last 40 cm (dribble / shoot).

State machine (all positions in the `map` frame):
  SEARCH    ball not seen            -> spin in place
  APPROACH  ball seen, robot not lined up behind it
            -> Nav2 goal "behind the ball, facing the aim point" (Nav2 costmaps avoid the opponent).
               If the robot is on the wrong side of the ball it first goes around via a detour point.
  DRIBBLE   robot is behind the ball on the ball->aim line
            -> Stanley-style steering keeps robot on that line and pushes the ball toward the aim point.
               The opponent (lidar points that are NOT on the static map walls) repels the heading.
  SHOOT     close to the goal, aligned -> short full-speed burst through the ball

Final destination ("상대방 최종 목적지 좌표"):
  default = centre of the +X goal line (goal_x, goal_y);  click a point in RViz (Publish Point) to change it
  at runtime;  `side_aim` shifts the aim to the side away from the goalkeeper.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros

from pinky_soccer_sim.gz_utils import wrap, yaw_from_quat
from pinky_soccer_sim.field import default_dims

from pinky_soccer_game.avoidance import PrepareRouter, avoid_opponent, detour_target, goal_mouth_guard

# defaults = the default field (pinky_soccer_sim/config/field_hardboard.yaml); launch files pass the real values
FIELD_L, FIELD_W, FIELD_G, FIELD_STRIP = default_dims()


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class Striker(Node):
    def __init__(self):
        super().__init__('striker')
        P = self.declare_parameter
        P('field_length', FIELD_L)
        P('field_width', FIELD_W)
        P('goal_width', FIELD_G)
        P('goal_x', float('nan'))          # nan -> field_length/2
        P('goal_y', 0.0)
        P('attack_yaw', 0.0)               # direction that points INTO the goal (+X)
        P('v_dribble', 0.25)
        P('v_shoot', 0.65)                 # velocity_smoother caps at 0.70 m/s
        P('w_max', 3.0)
        P('k_lateral', 5.0)
        P('behind_dist', 0.08)             # keep the stock robot nose close enough to push the ball
        P('shoot_range', 0.90)             # start shooting when ball is this close to the goal
        P('shoot_lat', 0.06)               # [m] max offset from the ball->aim line to shoot
        P('shoot_heading', 0.35)           # [rad] max heading error to shoot (20 deg)
        P('dribble_lat', 0.10)             # [m] max offset from the line to START a dribble (0.15 to keep it)
        P('prepare_during_goal', True)     # the other team celebrates: walk back to our start pose meanwhile
        P('ball_timeout', 1.0)
        P('opp_range', 0.35)               # opponent repulsion radius
        P('side_aim', True)
        P('base_frame', 'base_footprint')
        P('team', 'yellow')                # yellow | blue (kickoff turns; the attack side is goal_x/attack_yaw)
        P('cmd_topic', 'cmd_vel_nav')     # -> nav2 velocity_smoother -> /cmd_vel (single publisher chain)
        P('ball_hold_x', 0.07)             # ball centre just ahead of base_footprint during a push
        P('ball_hold_time', 1.0)           # [s] short grace period for brief camera occlusion
        P('wall_margin', 0.12)
        P('wall_guard_margin', float('nan'))   # dribble wall guard (nan = 0.6 * wall_margin); < 0 lets the robot
                                               # run along a wall pushing a wall ball
        P('enable_nav', True)
        P('start_delay', 6.0)              # wait for Nav2 / AMCL
        P('match_control', False)
        P('direct_approach_range', 0.6)    # [m] behind-the-ball point closer than this -> direct control
        P('v_approach', 0.45)              # direct approach speed (= Nav2 desired_linear_vel)
        P('stuck_time', 1.2)               # [s] neither moving nor turning (jammed) -> back off
        P('jam_dist', 0.22)                # [m] lidar obstacle this close counts as 'pressed against something'
        P('backoff_dist', 0.05)            # [m] how far to back off before searching the ball again
        P('prepare_direct_range', 0.30)    # PREPARE: closer than this to the ready pose -> direct control
        P('ready_offset', 0.38)            # [m] start pose = own goal line + this (real robot: back of the
                                           # robot 5 cm from its goal line -> 0.05 + 0.075 = 0.125)
        P('orbit_radius', 0.24)            # [m] circle around the ball when we are on the wrong side of it
                                           # (real robot: 0.32 = wide loop, room to line up for the shot)
        P('celebrate', True)               # after OUR goal: drive between own goal and centre line, spin
        P('celebrate_turns', 5.0)          # full turns on the spot
        P('celebrate_w', 3.0)              # [rad/s] spin speed (real wheels allow ~5.9)
        P('use_opponent', True)            # False: no opponent on the field (real single-robot run) -> lidar
                                           # points that do not match the map are NOT dodged
        P('search_back_dist', 0.15)        # [m] ball lost right in front: back off this far to see it again
        P('search_w', 1.5)                 # [rad/s] spin while looking for the ball (real robot: slow, 0.6)
        P('lost_backup_time', 2.0)         # [s] ball vanished right in front -> back up to see it, not spin
        P('outer_strip', FIELD_STRIP)      # field edge -> wall distance (detour points may use it)

        L = self.get_parameter('field_length').value
        gx = self.get_parameter('goal_x').value
        self.L = L
        self.W = self.get_parameter('field_width').value
        self.G = self.get_parameter('goal_width').value
        self.goal = np.array([L / 2 if math.isnan(gx) else gx, self.get_parameter('goal_y').value])
        self.attack_yaw = self.get_parameter('attack_yaw').value
        strip = self.get_parameter('outer_strip').value
        self.router = PrepareRouter(L / 2 - 0.12, self.W / 2 + strip - 0.09, stuck_time=1.5)  # x: out of the goals

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ball = None
        self.ball_t = 0.0
        self.opp = None
        self.opp_t = 0.0
        self.map = None
        self.scan = None
        self.state = 'SEARCH'
        self.shoot_until = 0.0
        self.nav_handle = None
        self.nav_goal_xy = None
        self.nav_goal_t = 0.0
        self.nav_request_id = 0
        self.nav_request_pending = False
        self.t_start = time.time()
        self.last_rel = None
        self.last_seen = 0.0
        self.match_control = self.get_parameter('match_control').value
        self.match_phase = 'WAIT' if self.match_control else 'PLAY'
        self.ready_goal_sent = False
        self.kickoff = 'OPEN'
        self.stuck_ref = None
        self.backoff_until = 0.0
        self.backoff_v = -0.10
        self.orbit_dir = None              # +1 / -1 while circling the ball (kept until lined up)
        self.search_phase = None           # ball lost: back -> spin -> roam (see search())
        self.search_wp = -1
        self.search_until = 0.0
        self.search_turned = self.search_yaw = 0.0
        self.search_target = np.zeros(2)
        self.celebrating = False           # our goal: celebration running during the GOAL phase
        self.goal_t = 0.0
        self.spin_done = 0.0               # [rad] turned so far
        self.spin_last = None

        self.create_subscription(PointStamped, 'ball/position_map', self._ball, 10)
        self.create_subscription(LaserScan, 'scan', self._scan, qos_profile_sensor_data)
        self.create_subscription(
            OccupancyGrid, 'map', self._map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointStamped, '/clicked_point', self._clicked, 10)
        if self.match_control:
            self.create_subscription(String, '/game/control', self._match_control, 10)
            self.create_subscription(String, '/game/kickoff', self._kickoff, 10)
            self.create_subscription(String, '/game/goal', self._goal_scored, 10)

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)
        self.state_pub = self.create_publisher(String, 'game/state', 10)   # /<ns>/game/state
        self.opp_pub = self.create_publisher(PointStamped, 'opponent/position_map', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'striker ready, goal target = {self.goal.tolist()} (click in RViz to change)')

    # ------------------------------------------------------------------ subscriptions
    def _ball(self, m):
        self.ball = np.array([m.point.x, m.point.y])
        self.ball_t = time.time()

    def _scan(self, m):
        self.scan = m

    def _map(self, m):
        self.map = (np.array(m.data, dtype=np.int8).reshape(m.info.height, m.info.width),
                    m.info.resolution, m.info.origin.position.x, m.info.origin.position.y)

    def _clicked(self, m):
        self.goal = np.array([m.point.x, m.point.y])
        self.get_logger().info(f'new goal target: {self.goal.tolist()}')

    def _kickoff(self, msg):
        self.kickoff = msg.data.strip().upper()

    def _match_control(self, msg):
        if msg.data == 'PREPARE':
            self.celebrating = False
            self.match_phase = 'PREPARE'
            self.ready_goal_sent = False
            self.router.reset()
            self.cancel_nav()
            self.get_logger().info('preparing at the yellow goal')
        elif msg.data == 'START':
            self.match_phase = 'PLAY'
            self.t_start = time.time() - self.get_parameter('start_delay').value
            self.cancel_nav()
            self.get_logger().info('match started')
        elif msg.data == 'GOAL':
            self.match_phase = 'GOAL'
            self.goal_t = time.time()
            self.cancel_nav()
            self.spin_done, self.spin_last = 0.0, None
        elif msg.data == 'WAIT':
            self.celebrating = False
            self.match_phase = 'WAIT'
            self.cancel_nav()

    def _goal_scored(self, msg):
        """/game/goal: the team that just scored (sent by the referee right before GOAL)."""
        self.celebrating = self.get_parameter('celebrate').value and \
            msg.data.strip().upper() == self.get_parameter('team').value.upper()
        self.spin_done, self.spin_last = 0.0, None

    def celebrate(self, rx, ry, ryaw):
        """Our goal: go to the middle of our own half (between own goal and centre line), spin
        `celebrate_turns` times on the spot, then report CELEBRATE_DONE (the referee waits for it)."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        spot = (self.goal - n * self.L) + n * (self.L / 4)               # own goal line -> quarter of the field
        dist = float(np.linalg.norm(spot - np.array([rx, ry])))
        target = self.get_parameter('celebrate_turns').value * 2 * math.pi
        if self.spin_last is None and dist > 0.06:
            # direct steering (the other robot stands still during GOAL; Nav2 goals kept being re-sent/cancelled)
            self.set_state('CELEBRATE_GO')
            self.cancel_nav()
            opp = self.estimate_opponent((rx, ry, ryaw))
            self.play_ctx = ((rx, ry, ryaw), opp, None)
            self.publish_play(self.direct_goto(rx, ry, ryaw, spot, self.attack_yaw, pos_tol=0.05,
                                               vmax=self.get_parameter('v_approach').value))
            return
        self.cancel_nav()
        if self.spin_last is None:
            self.spin_last = ryaw
        self.spin_done += abs(wrap(ryaw - self.spin_last))
        self.spin_last = ryaw
        cmd = Twist()
        if self.spin_done < target:
            self.set_state('CELEBRATE_SPIN')
            cmd.angular.z = self.get_parameter('celebrate_w').value
        else:
            self.set_state('CELEBRATE_DONE')
        self.cmd_pub.publish(cmd)

    # ------------------------------------------------------------------ helpers
    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', self.get_parameter('base_frame').value,
                                                rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        return (t.transform.translation.x, t.transform.translation.y, yaw_from_quat(t.transform.rotation))

    def on_static_map(self, x, y):
        if self.map is None:
            return False
        grid, res, ox, oy = self.map
        c, r = int((x - ox) / res), int((y - oy) / res)
        h, w = grid.shape
        if not (0 <= c < w and 0 <= r < h):
            return True                                   # outside the map: not an opponent
        r0, r1, c0, c1 = max(0, r - 2), min(h, r + 3), max(0, c - 2), min(w, c + 3)
        return bool((grid[r0:r1, c0:c1] > 50).any())

    def estimate_opponent(self, pose):
        """Lidar returns that are not explained by the static map -> opponent robot (nearest cluster)."""
        if not self.get_parameter('use_opponent').value:
            return None
        if self.scan is None or self.map is None or pose is None:
            return self.opp if time.time() - self.opp_t < 0.6 else None
        s = self.scan
        try:    # use the TF of the scan frame: Pinky's rplidar_link is rotated 180 deg w.r.t. base_link
            t = self.tf_buffer.lookup_transform('map', s.header.frame_id, rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        lx, ly = t.transform.translation.x, t.transform.translation.y
        lyaw = yaw_from_quat(t.transform.rotation)
        ranges = np.array(s.ranges, dtype=float)
        ang = s.angle_min + np.arange(len(ranges)) * s.angle_increment
        ok = np.isfinite(ranges) & (ranges > 0.10) & (ranges < 1.8)
        pts = []
        for r, a in zip(ranges[ok], ang[ok]):
            px, py = r * math.cos(a), r * math.sin(a)
            mx = lx + math.cos(lyaw) * px - math.sin(lyaw) * py
            my = ly + math.sin(lyaw) * px + math.cos(lyaw) * py
            if not self.on_static_map(mx, my):
                pts.append((mx, my, r))
        if pts:
            pts.sort(key=lambda p: p[2])
            nx, ny, _ = pts[0]
            cl = [(x, y) for x, y, _ in pts if math.hypot(x - nx, y - ny) < 0.15]
            if len(cl) >= 2:
                self.opp = np.array([np.mean([c[0] for c in cl]), np.mean([c[1] for c in cl])])
                self.opp_t = time.time()
        if time.time() - self.opp_t < 0.6 and self.opp is not None:
            m = PointStamped()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.point.x, m.point.y = float(self.opp[0]), float(self.opp[1])
            self.opp_pub.publish(m)
            return self.opp
        return None

    def aim_point(self, opp):
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat = np.array([-n[1], n[0]])
        aim = self.goal.copy()
        if self.get_parameter('side_aim').value and opp is not None \
                and np.linalg.norm(opp - self.goal) < 0.7:
            s = float(np.dot(opp - self.goal, lat))
            aim = self.goal - lat * math.copysign(min(self.G / 2 - 0.10, 0.18), s)
        return aim

    def inside_field(self, p, margin):
        """Rough interior test assuming the goal defines the +X axis of the field."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat = np.array([-n[1], n[0]])
        d = p - self.goal
        along, side = float(np.dot(d, n)), float(np.dot(d, lat))
        in_mouth = abs(side) < self.G / 2 - 0.05
        return (along < -margin or in_mouth) and abs(side) < self.W / 2 - margin

    # ------------------------------------------------------------------ nav2
    def send_nav(self, xy, yaw):
        if not self.get_parameter('enable_nav').value:
            return
        if not self.nav.server_is_ready():
            self.nav.wait_for_server(timeout_sec=0.01)
            return
        now = time.time()
        if self.nav_goal_xy is not None and np.linalg.norm(xy - self.nav_goal_xy) < 0.06 \
                and (self.nav_request_pending or self.nav_handle is not None):
            return
        if self.nav_goal_xy is not None:
            self.cancel_nav()
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.nav_goal_xy, self.nav_goal_t = xy.copy(), now
        self.nav_request_pending = True
        self.nav_request_id += 1
        request_id = self.nav_request_id
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(lambda future: self._nav_accepted(future, request_id))

    def _nav_accepted(self, fut, request_id):
        try:
            handle = fut.result()
        except Exception as exc:
            self.get_logger().error(f'failed to receive Nav2 goal response: {exc}')
            if request_id == self.nav_request_id:
                self.nav_request_pending = False
                self.nav_goal_xy = None
            return
        if request_id != self.nav_request_id:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        self.nav_request_pending = False
        if not handle.accepted:
            self.get_logger().warning('Nav2 rejected the navigation goal')
            self.nav_goal_xy = None
            return
        self.nav_handle = handle
        handle.get_result_async().add_done_callback(
            lambda future: self._nav_finished(future, request_id))

    def _nav_finished(self, fut, request_id):
        if request_id != self.nav_request_id:
            return
        try:
            fut.result()
        except Exception as exc:
            self.get_logger().error(f'Nav2 navigation result failed: {exc}')
        self.nav_handle = None
        self.nav_goal_xy = None

    def cancel_nav(self):
        self.nav_request_id += 1
        self.nav_request_pending = False
        if self.nav_handle is not None:
            try:
                self.nav_handle.cancel_goal_async()
            except Exception:
                pass
            self.nav_handle = None
        self.nav_goal_xy = None

    def direct_goto(self, rx, ry, ryaw, target, target_yaw, pos_tol=0.04, vmax=0.25):
        """Position first (forward or reverse, whichever needs less turning), then turn to target_yaw."""
        cmd = Twist()
        dx, dy = float(target[0] - rx), float(target[1] - ry)
        dist = math.hypot(dx, dy)
        if dist > pos_tol:
            err = wrap(math.atan2(dy, dx) - ryaw)
            direction = 1.0
            if abs(err) > math.pi / 2:
                err, direction = wrap(err + math.pi), -1.0
            wm = min(2.5, self.get_parameter('w_max').value)
            cmd.angular.z = clip(2.5 * err, -wm, wm)
            if abs(err) < 0.8:
                cmd.linear.x = direction * min(vmax, 2.0 * dist) * math.cos(err) ** 2
        else:
            wm = min(2.5, self.get_parameter('w_max').value)
            cmd.angular.z = clip(2.5 * wrap(target_yaw - ryaw), -wm, wm)
        return cmd

    # ------------------------------------------------------------------ main loop
    def stop(self):
        self.cmd_pub.publish(Twist())

    def tick(self):
        if self.match_control and self.match_phase == 'GOAL' and self.celebrating:
            pose = self.robot_pose()
            if pose is not None:
                self.celebrate(*pose)
                return
        if self.match_control and self.match_phase == 'GOAL' and not self.celebrating \
                and self.get_parameter('prepare_during_goal').value and time.time() - self.goal_t > 0.5:
            # the other team is celebrating: walk back to our start pose now instead of standing idle
            # (0.5 s grace: /game/goal may arrive just after GOAL - the scorer must not leave)
            self.match_phase = 'PREPARE'
            self.router.reset()
            self.cancel_nav()
        if self.match_control and self.match_phase in ('WAIT', 'GOAL'):
            self.stop()
            self.set_state(self.match_phase)
            return
        if time.time() - self.t_start < self.get_parameter('start_delay').value:
            return
        pose = self.robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        if self.match_control and self.match_phase in ('PREPARE', 'READY'):
            # in front of our own goal, facing the centre spot (where the ball is put back)
            n_att = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
            own_goal = self.goal - n_att * self.L
            ready_pose = own_goal + n_att * self.get_parameter('ready_offset').value
            ready_yaw = self.attack_yaw
            dist = float(np.linalg.norm(np.array([rx, ry]) - ready_pose))
            if dist < 0.05 and abs(wrap(ryaw - ready_yaw)) < 0.15:
                self.cancel_nav()
                self.stop()
                self.match_phase = 'READY'
                self.set_state('READY')
            elif self.match_phase == 'PREPARE':
                opp = self.estimate_opponent(pose)
                route = self.router.update((rx, ry, ryaw), ready_pose, opp, time.time(),
                                           moving_wanted=dist > 0.05)
                if route[0] == 'backoff':
                    # pushed into the other robot: get off it first, then take the wide way round
                    self.cancel_nav()
                    cmd = Twist()
                    cmd.linear.x = route[1]
                    cmd.linear.x = goal_mouth_guard((rx, ry, ryaw), cmd.linear.x, self.L / 2, self.G)
                    self.cmd_pub.publish(cmd)
                    self.set_state('PREPARE_BACKOFF')
                    return
                _, goal_xy, is_via = route
                if is_via:
                    # going round the other robot (Nav2's global plan only knows the walls)
                    self.cancel_nav()
                    via = np.array(goal_xy)
                    face = math.atan2(ready_pose[1] - via[1], ready_pose[0] - via[0])
                    self.play_ctx = ((rx, ry, ryaw), opp, None)
                    self.publish_play(self.direct_goto(rx, ry, ryaw, via, face, pos_tol=0.0,
                                                       vmax=self.get_parameter('v_approach').value))
                    self.set_state('PREPARE_DETOUR')
                    return
                near_opp = opp is not None and math.hypot(opp[0] - rx, opp[1] - ry) < 0.40
                if near_opp:
                    # other robot close: direct control through the collision filter (Nav2 would brush it)
                    self.cancel_nav()
                    self.play_ctx = ((rx, ry, ryaw), opp, None)
                    self.publish_play(self.direct_goto(rx, ry, ryaw, ready_pose, ready_yaw, pos_tol=0.04,
                                                       vmax=self.get_parameter('v_approach').value))
                elif dist > self.get_parameter('prepare_direct_range').value:
                    self.send_nav(ready_pose, ready_yaw)
                else:
                    # Short final move: Nav2 (RPP) cannot reverse, so a target just behind the robot (e.g. the
                    # spawn pose is 0.165 m in front of the ready pose) costs two 180 deg turns plus a wobble
                    # around the tiny replanned path.  Drive there directly, backing up if that is shorter.
                    self.cancel_nav()
                    self.cmd_pub.publish(self.direct_goto(rx, ry, ryaw, ready_pose, ready_yaw))
                self.set_state('PREPARING')
            else:
                self.stop()
                self.set_state('PREPARING')
            return

        me_team = self.get_parameter('team').value.upper()
        if self.match_control and self.kickoff in ('YELLOW', 'BLUE') and self.kickoff != me_team:
            # the other team's kickoff: it must touch the ball first -> stay at the start position
            self.cancel_nav()
            self.stop()
            self.set_state('KICKOFF_WAIT')
            return

        r = np.array([rx, ry])
        opp = self.estimate_opponent(pose)
        self.play_ctx = ((rx, ry, ryaw), opp, None)

        now = time.time()
        # jammed (last resort - normally the other robot is avoided as an obstacle): neither driving nor turning
        # gets anywhere -> back off ~5 cm away from the obstacle, then carry on (the ball stays remembered)
        if now < self.backoff_until:
            self.cancel_nav()
            cmd = Twist()
            cmd.linear.x = self.backoff_v
            self.cmd_pub.publish(cmd)                 # not through the collision filter: we move away
            self.set_state('BACKOFF')
            return
        ref = self.stuck_ref
        if ref is None or math.hypot(rx - ref[0], ry - ref[1]) > 0.02 or abs(wrap(ryaw - ref[2])) > 0.1 \
                or self.state in ('NO_ROOM', 'KICKOFF_WAIT'):           # standing still on purpose
            self.stuck_ref = (rx, ry, ryaw, now)
        elif now - ref[3] > self.get_parameter('stuck_time').value:
            near = self.nearest_obstacle()
            ball_far = self.ball is None or float(np.linalg.norm(self.ball - r)) > 0.12
            if (near is not None and near[0] < self.get_parameter('jam_dist').value) or ball_far:
                dist = self.get_parameter('backoff_dist').value
                speed = 0.10
                front = near is None or near[1]
                self.backoff_v = -speed if front else speed  # away from whatever we are pressed against
                self.backoff_until = now + dist / speed
                self.get_logger().info(f'jammed -> {dist * 100:.0f} cm back'
                                       f'{" (obstacle %.2f m)" % near[0] if near else ""}')
            self.stuck_ref = None
        seen = self.ball is not None and now - self.ball_t <= self.get_parameter('ball_timeout').value
        if seen:
            b = self.ball
            c, sn = math.cos(ryaw), math.sin(ryaw)
            dv = b - r
            self.last_rel = (c * dv[0] + sn * dv[1], -sn * dv[0] + c * dv[1])    # ball in the robot frame
            self.last_seen = now
        elif self.match_control and self.kickoff == me_team:
            # our kickoff: the referee put the ball 5 cm in front of the bumper - below the camera's view
            # at that distance, so push straight ahead where it must be
            b = r + 0.12 * np.array([math.cos(ryaw), math.sin(ryaw)])
        elif self.state in ('DRIBBLE', 'SHOOT') and self.last_rel is not None \
                and 0.03 < self.last_rel[0] < 0.30 and abs(self.last_rel[1]) < 0.12 \
                and now - self.last_seen < self.get_parameter('ball_hold_time').value:
            # Keep pushing briefly if the ball is momentarily hidden by the robot body.
            b = r + self.get_parameter('ball_hold_x').value * np.array([math.cos(ryaw), math.sin(ryaw)])
        else:
            self.cancel_nav()
            self.search(rx, ry, ryaw, now)
            return
        self.search_phase = None                      # ball (re)found: the next loss starts a fresh search

        self.play_ctx = ((rx, ry, ryaw), opp, (float(b[0]), float(b[1])))
        if abs(b[0]) > self.L / 2 + 0.01 and abs(b[1]) > self.G / 2:
            # ball behind a goal line but beside the goal: out of play, cannot be scored -> wait for the
            # referee / a person to put it back instead of circling it next to the wall
            self.cancel_nav()
            self.stop()
            self.set_state('OUT')
            return

        aim = self.shootable_aim(b, self.aim_point(opp))
        if aim is None:
            # boxed in (corner): no spot behind the ball for any aim - stand off on the field side facing it,
            # do not circle forever; the referee's corner rule restarts
            self.set_state('NO_ROOM')
            c = -b / (float(np.linalg.norm(b)) + 1e-6)
            R = self.get_parameter('orbit_radius').value
            wait_pt = b + c * R
            face = math.atan2(b[1] - wait_pt[1], b[0] - wait_pt[0])
            self.cancel_nav()
            self.publish_play(self.direct_goto(rx, ry, ryaw, wait_pt, face, pos_tol=0.04,
                                               vmax=self.get_parameter('v_approach').value))
            return
        to_aim = aim - b
        dist_aim = float(np.linalg.norm(to_aim))
        if dist_aim < 0.03:
            self.stop()
            return
        u = to_aim / dist_aim
        line_ang = math.atan2(u[1], u[0])
        d = r - b
        along = float(np.dot(u, d))                       # < 0  : robot is behind the ball
        lat = float(u[0] * d[1] - u[1] * d[0])            # > 0  : robot is left of the line
        w_max = self.get_parameter('w_max').value

        # ---- SHOOT burst in progress
        if time.time() < self.shoot_until:
            err = wrap(line_ang - ryaw)
            cmd = Twist()
            cmd.linear.x = self.get_parameter('v_shoot').value
            cmd.angular.z = clip(3.0 * err, -w_max, w_max)
            self.publish_play(cmd)
            return

        lat_limit = 0.15 if self.state == 'DRIBBLE' else self.get_parameter('dribble_lat').value
        behind = -0.40 < along < -0.03 and abs(lat) < lat_limit

        if behind:
            self.set_state('DRIBBLE')
            self.cancel_nav()
            heading = line_ang - math.atan(self.get_parameter('k_lateral').value * lat)

            # opponent repulsion
            slow = 1.0
            if opp is not None:
                vec = opp - r
                d_opp = float(np.linalg.norm(vec))
                rng = self.get_parameter('opp_range').value
                rel = wrap(math.atan2(vec[1], vec[0]) - ryaw)
                if d_opp < rng and abs(rel) < 1.2:
                    heading += -math.copysign(0.9, rel) * (1.0 - d_opp / rng)
                    slow = clip(d_opp / rng + 0.3, 0.3, 1.0)

            # wall guard: do not drive the ball (or ourselves) into a wall
            probe = r + 0.16 * np.array([math.cos(ryaw), math.sin(ryaw)])
            guard = self.get_parameter('wall_guard_margin').value
            if math.isnan(guard):
                guard = self.get_parameter('wall_margin').value * 0.6
            if not self.inside_field(probe, guard):
                centre = self.goal - np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)]) * self.L / 2
                heading = math.atan2(centre[1] - ry, centre[0] - rx)
                slow = 0.4

            err = wrap(heading - ryaw)
            cmd = Twist()
            cmd.angular.z = clip(2.5 * err, -w_max, w_max)
            cmd.linear.x = self.get_parameter('v_dribble').value * slow * max(0.0, math.cos(err)) ** 2 \
                if abs(err) < 1.0 else 0.0

            # shoot?
            if dist_aim < self.get_parameter('shoot_range').value and abs(lat) < self.get_parameter('shoot_lat').value \
                    and abs(wrap(line_ang - ryaw)) < self.get_parameter('shoot_heading').value and along > -0.25 \
                    and (opp is None or np.linalg.norm(opp - r) > 0.25):
                self.set_state('SHOOT')
                self.shoot_until = time.time() + 0.7
            self.publish_play(cmd)
            return

        # ---- wrong side of the ball (e.g. between the ball and the goal, no room to shoot): circle it widely
        if along > -0.10 or abs(lat) > 0.25:
            self.set_state('ORBIT')
            p = self.orbit_point(b, u, r)
            vmax = self.get_parameter('v_approach').value
            if float(np.linalg.norm(p - r)) < self.get_parameter('direct_approach_range').value:
                self.cancel_nav()
                self.publish_play(self.direct_goto(rx, ry, ryaw, p, line_ang, pos_tol=0.0, vmax=vmax))
            else:
                self.send_nav(p, line_ang)
            return
        self.orbit_dir = None

        # ---- APPROACH: behind the ball, facing the aim point
        self.set_state('APPROACH')
        behind_pt = b - u * self.get_parameter('behind_dist').value
        m = self.get_parameter('wall_margin').value
        n_att = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat_ax = np.array([-n_att[1], n_att[0]])
        dd = behind_pt - self.goal
        a_, s_ = float(np.dot(dd, n_att)), float(np.dot(dd, lat_ax))
        s_ = clip(s_, -(self.W / 2 - m), self.W / 2 - m)
        a_ = min(a_, -m) if abs(s_) > self.G / 2 - 0.05 else a_
        a_ = max(a_, -(self.L - m))
        behind_pt = self.goal + n_att * a_ + lat_ax * s_
        if float(np.linalg.norm(behind_pt - r)) < self.get_parameter('direct_approach_range').value:
            # close to the ball: drive there directly (like the blue attacker) instead of waiting for Nav2
            # replanning + rotate-to-heading, which lost every close race for the ball
            self.cancel_nav()
            # the other robot between us and that point -> go past its side first (no body contact)
            tgt, side_step = detour_target((rx, ry, ryaw), behind_pt, opp, self.router)
            if side_step:
                self.set_state('AVOID')
                self.publish_play(self.direct_goto(rx, ry, ryaw, np.array(tgt), line_ang, pos_tol=0.0,
                                                   vmax=self.get_parameter('v_approach').value))
                return
            self.publish_play(self.direct_goto(rx, ry, ryaw, behind_pt, line_ang, pos_tol=0.03,
                                                  vmax=self.get_parameter('v_approach').value))
            return
        self.send_nav(behind_pt, line_ang)

    def shootable_aim(self, b, aim):
        """The spot behind the ball for this aim may lie beyond a wall (ball along a side wall near the goal):
        then no lining up is possible.  Try the posts, the front of the goal, finally clear the ball along the
        wall back into the field - the first aim whose spot behind the ball the robot can stand on."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat = np.array([-n[1], n[0]])
        post = self.G / 2 - 0.05
        deep = n * 0.10           # aim 10 cm behind the goal line: a clear direction even with the ball ON the line
        cands = [aim + deep, self.goal + deep + lat * post, self.goal + deep - lat * post, self.goal - n * 0.35,
                 b + n * 0.5,                                   # ball at a side wall: push it ALONG the wall
                 b - n * 0.5]                                   # last: back along the wall, out of the corner
        for a in cands:
            d = a - b
            dn = float(np.linalg.norm(d))
            if dn < 0.03:
                continue
            spot = b - d / dn * 0.18
            if self._reachable(spot):
                return a
            # next to a wall the exact spot is a few cm inside the wall (the robot is wider than the gap):
            # standing a little to the field side still slides the ball along the wall
            if float(np.linalg.norm(self._clamp_inside(spot) - spot)) < 0.07 and self._reachable(
                    self._clamp_inside(spot)):
                return a
        return None

    def _clamp_inside(self, p):
        """Nearest point where the robot centre can stand (see _reachable), field coordinates kept."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat = np.array([-n[1], n[0]])
        d = p - self.goal
        along, side = float(np.dot(d, n)), float(np.dot(d, lat))
        strip = self.get_parameter('outer_strip').value
        lim = self.W / 2 + strip - 0.095
        along = min(max(along, -(self.L + strip - 0.095)), -0.045)
        side = min(max(side, -lim), lim)
        return self.goal + n * along + lat * side

    def search(self, rx, ry, ryaw, now):
        """Ball lost.  1) it was right in front (under the camera): back up ~15 cm and look again;
        2) one full turn on the spot; 3) still nothing (hidden behind the other robot, far away): walk to
        the next look-out point (centre, then the middle of each quarter) and turn there once more ..."""
        cmd = Twist()
        if self.search_phase is None:
            rel = self.last_rel
            close = rel is not None and 0.0 < rel[0] < 0.25 and abs(rel[1]) < 0.15
            self.search_phase = 'back' if close else 'spin'
            self.search_until = now + self.get_parameter('search_back_dist').value / 0.10
            self.search_turned, self.search_yaw = 0.0, ryaw
        if self.search_phase == 'back':
            if now < self.search_until:
                self.set_state('LOOK_BACK')
                cmd.linear.x = -0.10
                self.publish_play(cmd)
                return
            self.search_phase, self.search_turned, self.search_yaw = 'spin', 0.0, ryaw
        if self.search_phase == 'spin':
            self.search_turned += abs(wrap(ryaw - self.search_yaw))
            self.search_yaw = ryaw
            if self.search_turned < 2 * math.pi:
                self.set_state('SEARCH')
                cmd.angular.z = self.get_parameter('search_w').value
                self.publish_play(cmd)
                return
            # a whole turn and nothing: go and look from somewhere else
            pts = [(0.0, 0.0), (self.L / 4, self.W / 4), (self.L / 4, -self.W / 4),
                   (-self.L / 4, -self.W / 4), (-self.L / 4, self.W / 4)]
            for _ in range(len(pts)):
                self.search_wp = (self.search_wp + 1) % len(pts)
                if math.hypot(pts[self.search_wp][0] - rx, pts[self.search_wp][1] - ry) > 0.25:
                    break
            self.search_target = np.array(pts[self.search_wp])
            self.search_phase, self.search_until = 'roam', now + 12.0
        if self.search_phase == 'roam':
            t = self.search_target
            if math.hypot(t[0] - rx, t[1] - ry) < 0.10 or now > self.search_until:
                self.search_phase, self.search_turned, self.search_yaw = 'spin', 0.0, ryaw   # look around there
                return
            self.set_state('ROAM')
            self.publish_play(self.direct_goto(rx, ry, ryaw, t, math.atan2(-t[1], -t[0]) if t.any() else ryaw,
                                               pos_tol=0.08, vmax=self.get_parameter('v_approach').value))

    def nearest_obstacle(self):
        """Closest lidar return in the robot frame: (distance from the robot centre, in front?).  The Pinky's
        rplidar_link is turned 180 deg (scan angle 0 = robot rear), 1.7 cm behind the robot centre."""
        s = self.scan
        if s is None:
            return None
        r = np.asarray(s.ranges, dtype=float)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment + math.pi
        ok = np.isfinite(r) & (r > max(0.05, s.range_min))
        if not ok.any():
            return None
        x, y = r[ok] * np.cos(a[ok]) - 0.017, r[ok] * np.sin(a[ok])
        i = int(np.argmin(np.hypot(x, y)))
        return float(np.hypot(x[i], y[i])), bool(x[i] > 0)

    def _reachable(self, p):
        """Robot centre can stand there: inside the walls (not inside a goal)."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        lat = np.array([-n[1], n[0]])
        d = p - self.goal
        along, side = float(np.dot(d, n)), float(np.dot(d, lat))
        strip = self.get_parameter('outer_strip').value
        return -(self.L + strip - 0.09) < along < -0.04 and abs(side) < self.W / 2 + strip - 0.09

    def orbit_point(self, b, u, r):
        """Next point on a circle (orbit_radius) around the ball, ~50 deg further toward the shooting spot
        behind the ball.  Short way round unless a wall blocks it; the chosen direction is kept until the robot
        is lined up, so it does not dither.  Near a wall the full circle may not fit (the spot behind the ball
        lies beyond the wall): then a smaller circle is used, down to ~15 cm."""
        R0 = self.get_parameter('orbit_radius').value
        back = -u
        v = r - b
        dist = float(np.linalg.norm(v))
        theta = math.atan2(back[0] * v[1] - back[1] * v[0], float(np.dot(back, v)))   # 0 = behind the ball
        short = -1.0 if theta > 0 else 1.0
        order = [self.orbit_dir, -self.orbit_dir] if self.orbit_dir else [short, -short]

        for R in (R0, 0.8 * R0, 0.6 * R0, max(0.15, 0.45 * R0)):
            def at(t, R=R):
                c, s_ = math.cos(t), math.sin(t)
                return b + R * np.array([back[0] * c - back[1] * s_, back[0] * s_ + back[1] * c])
            if not self._reachable(at(0.0)):          # the shooting spot itself does not fit: smaller circle
                continue
            if dist < 0.75 * R and self._reachable(at(theta)):   # too close: first straight out to the circle
                return at(theta)
            for direction in order:
                step = min(abs(theta), 0.9) if direction == short else 0.9
                p = at(theta + direction * step)
                if self._reachable(p):
                    self.orbit_dir = direction
                    return p
        c, s_ = math.cos(theta), math.sin(theta)
        return b + 0.15 * np.array([back[0] * c - back[1] * s_, back[0] * s_ + back[1] * c])
        short = -1.0 if theta > 0 else 1.0
        order = [self.orbit_dir, -self.orbit_dir] if self.orbit_dir else [short, -short]
        for direction in order:
            step = min(abs(theta), 0.9) if direction == short else 0.9
            p = at(theta + direction * step)
            if self._reachable(p):
                self.orbit_dir = direction
                return p
        return at(theta)

    def publish_play(self, cmd):
        """Every direct command during play goes through the collision filter (Nav2 avoids by itself)."""
        me, opp, ball = self.play_ctx
        cmd.linear.x, cmd.angular.z, _ = avoid_opponent(me, cmd.linear.x, cmd.angular.z, opp, ball,
                                                        w_max=self.get_parameter('w_max').value)
        cmd.linear.x = goal_mouth_guard(me, cmd.linear.x, self.L / 2, self.G)
        self.cmd_pub.publish(cmd)

    def set_state(self, s):
        if s != self.state:
            self.get_logger().info(f'state: {self.state} -> {s}')
            self.state = s
        self.state_pub.publish(String(data=self.state))


def main():
    rclpy.init()
    node = Striker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
