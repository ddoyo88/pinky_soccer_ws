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
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
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


class GuardedPublisher:
    """cmd_vel publisher with a last check (the white boundary line) applied to every command."""

    def __init__(self, pub, guard):
        self.pub, self.guard = pub, guard

    def publish(self, cmd):
        self.last_in = (cmd.linear.x, cmd.angular.z)
        out = self.guard(cmd)
        self.last_out = (out.linear.x, out.angular.z)
        self.pub.publish(out)


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
        P('celebrate_turns', 1.0)          # full turns on the spot (user, 2026-11-26: 3 was too long)
        P('concede_centre', True)          # the other team scored: go to the centre line (on our side, away
                                           # from the scorer) first, then back to our start pose - not standing
                                           # in front of our goal where the scorer is
        P('concede_hold', 1.5)             # [s] wait at the centre-line point before going back
        P('scorer_delay', 2.0)             # [s] after a restart the team that scored last waits this long
        P('first_kick', 'yellow')          # the match's first kickoff: this team goes, the other waits
                                           # scorer_delay (the same 3 s head start)
        P('celebrate_w', 3.0)              # [rad/s] spin speed (real wheels allow ~5.9)
        P('use_opponent', True)            # False: no opponent on the field (real single-robot run) -> lidar
                                           # points that do not match the map are NOT dodged
        P('search_back_dist', 0.15)        # [m] ball lost right in front: back off this far to see it again
        P('search_w', 1.5)                 # [rad/s] spin while looking for the ball (real robot: slow, 0.6)
        P('lost_backup_time', 2.0)         # [s] ball vanished right in front -> back up to see it, not spin
        P('outer_strip', FIELD_STRIP)      # field edge -> wall distance (detour points may use it)
        P('policy_path', '')               # RL policy (.npz from pinky_soccer_rl.export): plays the PLAY phase
                                           # instead of the rule-based state machine; '' = rules
        P('rl_idle_after', 3.0)            # [s] RL mode: the robot hardly moves this long in play -> the rule-
        P('rl_rules_for', 4.0)             # based striker plays this long (orbit, dribble), then the policy again
        P('ball_play', 'rules')              # 'rules': look - small move - look again (ball_play.py); '' = the
                                             # old state machine (the learned policy is not used any more)
        P('rl_search_after', 3.0)          # [s] RL mode: ball unseen this long -> the rule-based search (back up,
                                           # turn, roam) until the camera sees it again, then the policy again
        P('line_guard', True)              # never drive over the white boundary line (camera, line/boundary)
        P('robot_guard', True)             # never drive into the other robot (lidar): slow from 0.32 m centre
                                           # distance, stop at 0.20 m - same rule as the RL training
        P('robot_front', 0.05)             # [m] base_footprint -> bumper front
        P('robot_back', 0.08)              # [m] base_footprint -> rear
        P('robot_half_width', 0.065)       # [m]
        P('line_margin', 0.03)             # [m] stop this far before the line (+ braking distance)

        L = self.get_parameter('field_length').value
        gx = self.get_parameter('goal_x').value
        self.L = L
        self.W = self.get_parameter('field_width').value
        self.G = self.get_parameter('goal_width').value
        self.goal = np.array([L / 2 if math.isnan(gx) else gx, self.get_parameter('goal_y').value])
        self.attack_yaw = self.get_parameter('attack_yaw').value
        strip = self.get_parameter('outer_strip').value
        self.router = PrepareRouter(L / 2 - 0.12, self.W / 2 - 0.12, stuck_time=1.5)  # x: out of the goals
        # y: 0.12 m inside the white line - via points beyond it were blocked by the line guard, the router took
        # that for 'stuck' and looped backoff <-> detour for 40 s (final5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ball = None
        self.ball_t = 0.0
        self.rl_pp = {}                    # RL command post-processing state (rl_post_process)
        self.attack_msg, self.attack_t = None, 0.0     # /game/attack from the referee (attack/defence game)
        self.kick_circle = False
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
        self.concede_xy, self.concede_t0, self.concede_hold_t = None, 0.0, None
        self.scored_last, self.hold_until = False, 0.0
        self.first_kickoff = True
        self.stuck_ref = None
        self.backoff_until = 0.0
        self.jam_count, self.jam_last_t, self.escape_until = 0, 0.0, 0.0
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
            self.create_subscription(String, '/game/event', self._game_event, 10)
            self.create_subscription(String, '/game/attack', self._attack_msg, 10)
            other = 'blue' if self.get_parameter('team').value.lower() == 'yellow' else 'yellow'
            self.create_subscription(PointStamped, f'/{other}/ball/position_map', self._shared_ball, 10)

        self.cmd_pub = GuardedPublisher(self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10),
                                        lambda c: self._line_guard(self._robot_guard(self._kick_circle_guard(c))))   # line last: moving
                                                                                            # apart never crosses it
        self.line_pts, self.line_t, self.line_stops = np.zeros((0, 2)), 0.0, 0
        self.create_subscription(PoseArray, 'line/boundary', self._line, 10)
        self.rl, self.rl_t, self.rl_cmd = None, 0.0, (0.0, 0.0)
        self.guard_since, self.rl_escape_until = None, 0.0
        self.guard_hit = False
        path = os.path.expanduser(self.get_parameter('policy_path').value)
        if path:
            from pinky_soccer_game.rl_policy import RLPlayer
            self.rl = RLPlayer(path, self.L, self.W, self.attack_yaw)
            self.get_logger().info(f'RL policy {path} plays (attack yaw {self.attack_yaw:+.2f}, '
                                   f'opponent input {"on" if self.rl.pol.use_opponent else "off"})')
        # localisation guard: a pose outside the walls is impossible -> re-seed AMCL (relative topic: the
        # namespaced sim AMCL or the real robot's own in its domain)
        self.initpose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.reseed_t = 0.0
        self.state_pub = self.create_publisher(String, 'game/state', 10)   # /<ns>/game/state
        self.opp_pub = self.create_publisher(PointStamped, 'opponent/position_map', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'striker ready, goal target = {self.goal.tolist()} (click in RViz to change)')

    # ------------------------------------------------------------------ subscriptions
    def _ball(self, m):
        if time.time() < getattr(self, 'ball_reset_until', 0.0) and math.hypot(m.point.x, m.point.y) > 0.30:
            return                                        # a frame from before the referee moved the ball
        self.ball = np.array([m.point.x, m.point.y])
        self.ball_t = time.time()
        self.own_ball_t = self.ball_t
        self.carrying = False
        if time.time() >= getattr(self, 'ball_reset_until', 0.0):
            self.ball_placed_until = 0.0                  # a real sighting replaces the placed position

    def _shared_ball(self, m):
        """The other robot's camera estimate of the ball (relayed, map frame; ~5-10 cm off - two localisations).
        Used only while our own camera has not seen the ball for 0.5 s: we then know where to turn and look
        instead of spinning round.  Counted as 0.4 s old, so the look-again rule still turns to see it ourselves."""
        now = time.time()
        if now < getattr(self, 'ball_reset_until', 0.0) or getattr(self, 'carrying', False):
            return
        x, y = m.point.x, m.point.y
        if abs(x) > self.L / 2 + 0.15 or abs(y) > self.W / 2 + 0.15:
            return
        if now - getattr(self, 'own_ball_t', 0.0) > 0.5:
            self.ball = np.array([x, y])
            self.ball_t = now - 0.4
            self.shared_used = getattr(self, 'shared_used', 0) + 1

    def _attack_msg(self, m):
        self.attack_msg, self.attack_t = m.data.strip().upper(), time.time()

    def _role(self):
        """attack / defend / kickoff (go to the ball) / wait (the other team kicks off) / None (free game)."""
        a = self.attack_msg
        if a is None or a == 'FREE' or time.time() - self.attack_t > 2.0:
            return None
        team = self.get_parameter('team').value.upper()
        if a.endswith('_REGROUP'):
            # time-up swap: back to the own half first (the referee says when play goes on)
            return 'regroup' if a.startswith(team) else 'regroup_def'
        if a.endswith('_KICKOFF'):
            # the other team kicks off: we defend from the start (user: blue keeps its goal from the first
            # kick-off on)
            return 'kickoff' if a.startswith(team) else 'defend'
        return 'attack' if a == team else 'defend'

    def _game_event(self, m):
        """The referee put the ball back on the centre spot: forget the old spot (corner, beside the goal) and
        remember the ball at the centre, so the robot turns there at once instead of searching the old place."""
        t = m.data
        if t.startswith('새 경기 시작'):
            # a new match (also one restarted mid-game): first kick-off rules again (yellow first, blue +3 s)
            self.first_kickoff, self.scored_last, self.concede_xy = True, False, None
        if ('중앙' in t or '가운데' in t) and '못함' not in t:
            # the referee placed it there: a known position, drive to it straight away even if the camera does
            # not pick it up yet (attackers stood searching 15 s for a ball 0.7 m away on the centre spot, g01)
            self.ball = np.array([0.0, 0.0])
            self.ball_t = time.time()
            self.ball_reset_until = time.time() + 1.0
            self.ball_placed_until = time.time() + 8.0

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
            self.prepare_t0 = time.time()
            self.router.reset()
            self.cancel_nav()
            self.get_logger().info('preparing at the yellow goal')
        elif msg.data == 'START':
            self.match_phase = 'PLAY'
            if self.rl is not None:
                self.rl.reset()
            self.rl_pp = {}
            self.concede_xy = None
            first = self.first_kickoff and \
                self.get_parameter('team').value.lower() != self.get_parameter('first_kick').value.lower()
            # the robot that waits also keeps out of a 0.30 m circle round the ball until the other one has
            # played it (the kick-off was a face-off at the ball, 2026-11-25)
            self.kick_circle = bool(self.scored_last or first)
            self.kick_circle_t = time.time()
            if self.scored_last or first:     # we scored the last goal / not our first kickoff: head start
                self.hold_until = time.time() + self.get_parameter('scorer_delay').value
            self.scored_last = False
            self.first_kickoff = False
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
            self.concede_xy, self.scored_last, self.hold_until = None, False, 0.0
            self.first_kickoff = True                 # next START is a new match's first kickoff
            self.match_phase = 'WAIT'
            self.cancel_nav()

    def _goal_scored(self, msg):
        """/game/goal: the team that just scored (sent by the referee right before GOAL)."""
        mine = msg.data.strip().upper() == self.get_parameter('team').value.upper()
        self.celebrating = self.get_parameter('celebrate').value and mine
        self.scored_last = mine
        self.spin_done, self.spin_last = 0.0, None
        self.celebrate_back = None          # first: straight back out of the goal area (set on the first tick)
        self.celebrate_spot = None          # then: spin right there

    def celebrate(self, rx, ry, ryaw):
        """Our goal: go to the middle of our own half (between own goal and centre line), spin
        `celebrate_turns` times on the spot, then report CELEBRATE_DONE (the referee waits for it)."""
        # celebrate right where we are after backing out (rec2: crossing the field to the old spot in our own
        # half ran into the conceding robot on its way to its start pose - both stood for 40 s)
        if getattr(self, 'celebrate_spot', None) is None or self.spin_last is None and \
                getattr(self, 'celebrate_back', None) is None:
            self.celebrate_spot = None
        cb = getattr(self, 'celebrate_back', None)
        if self.celebrate_spot is None and not (cb and math.hypot(rx - cb[0], ry - cb[1]) < 0.15 and
                                                time.time() - cb[2] < 1.5) and cb is not None:
            self.celebrate_spot = np.array([rx, ry])
        spot = self.celebrate_spot if self.celebrate_spot is not None else np.array([rx, ry])
        dist = float(np.linalg.norm(spot - np.array([rx, ry])))
        target = self.get_parameter('celebrate_turns').value * 2 * math.pi
        # right after the goal: reverse straight out first (user, 2026-11-26) - 0.35 m back along our heading (we
        # face the goal we just scored in), no turning in front of it - then on to the celebration spot
        cb = getattr(self, 'celebrate_back', None)
        if cb is None and self.spin_last is None:
            near_goal = abs(rx - self.goal[0]) < 0.45
            self.celebrate_back = cb = ((rx, ry, time.time()) if near_goal else False)
        if cb and math.hypot(rx - cb[0], ry - cb[1]) < 0.15 and time.time() - cb[2] < 2.5:   # a little (user)
            self.set_state('CELEBRATE_BACK')
            self.cancel_nav()
            cmd = Twist()
            from pinky_soccer_game.rl_policy import over_line_ahead
            cmd.linear.x = 0.0 if over_line_ahead((rx, ry, ryaw), -0.10, self.L / 2, self.W / 2, self.G) else -0.10
            self.cmd_pub.publish(cmd)
            return
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

    def _unjam(self, pose, now):
        """Two robots pressed together, one against a line / in a corner: the robot guard blocks forward, the line
        guard backward - both stood still for 80 s (g06, 2026-11-26).  Not moved for 2 s with the other robot
        within 0.24 m -> turn away from it (bent toward the field centre if the line is that way) and drive
        out straight for 1.2 s."""
        x, y, yaw = pose
        uj = getattr(self, 'unjam', None)
        if uj is not None and now < uj[0]:
            err = wrap(uj[1] - yaw)
            cmd = Twist()
            cmd.angular.z = clip(2.0 * err, -0.8, 0.8)
            cmd.linear.x = 0.12 if abs(err) < 0.4 else 0.0
            self.cancel_nav()
            self.cmd_pub.publish(cmd)
            self.set_state('UNJAM')
            return True
        opp = self.estimate_opponent(pose)
        ref = getattr(self, 'unjam_ref', None)
        if ref is None or math.hypot(x - ref[0], y - ref[1]) > 0.03 or self.rl_pp.get('mode') in ('WAIT', 'KEEP'):
            self.unjam_ref = (x, y, now)
            return False
        if opp is None or math.hypot(opp[0] - x, opp[1] - y) > 0.23 or now - ref[2] < 2.0:
            return False
        from pinky_soccer_game.rl_policy import over_line_ahead
        away = math.atan2(y - opp[1], x - opp[0])
        centre = math.atan2(-y, -x)
        head = away
        for k in range(7):                                   # turn the escape direction toward the centre
            a = away + wrap(centre - away) * k / 6.0
            if not over_line_ahead((x, y, a), 0.12, self.L / 2, self.W / 2, self.G) and \
                    goal_mouth_guard((x, y, a), 0.12, self.L / 2, self.G) > 0.0:
                head = a
                break
        self.unjam = (now + 2.0, head)                       # ~0.8 s turning + ~1.2 s driving
        self.unjam_ref = None
        self.get_logger().info('pressed against the other robot: driving out')
        return False

    def _carry_ball(self, pose, now):
        """Camera blind spot: a ball right at the nose drops out of the image.  The remembered spot then stayed
        where the push began, the robot thought the ball was beside / behind it and went round or backed off
        to search (most of the 15 s standstills, g11, 2026-11-26).  Last seen just ahead and we kept driving
        forward -> the ball moves with the nose (for up to 2.5 s; a real sighting replaces it at once)."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        if self.ball is None:
            return
        fresh = now - self.ball_t < 0.25 and not getattr(self, 'carrying', False)
        if fresh:
            dx, dy = self.ball[0] - x, self.ball[1] - y
            self.ball_rel = (c * dx + s * dy, -s * dx + c * dy, now)
            return
        rel = getattr(self, 'ball_rel', None)
        if rel is None or now - rel[2] > 2.5 or not (0.0 < rel[0] < 0.22 and abs(rel[1]) < 0.08):
            self.carrying = False
            return
        v = getattr(self, 'rl_cmd', (0.0, 0.0))[0]
        if not getattr(self, 'carrying', False) and v <= 0.02:
            return                                       # not pushing: leave the memory as it is
        fx = min(rel[0], 0.075)                          # pushed: the ball sits against the bumper
        self.carrying = True
        self.ball = np.array([x + c * fx - s * rel[1], y + s * fx + c * rel[1]])
        self.ball_t = now - 0.3                          # remembered, not seen (keeps 'fresh' tests honest)

    def _rl_play(self, pose, now):
        """PLAY with the learned policy: a new action every 0.1 s (the training rate), repeated in between."""
        if self._unjam(pose, now):
            return
        self._carry_ball(pose, now)
        unseen = (now - self.ball_t) if self.ball is not None else 1e9
        if now < getattr(self, 'ball_placed_until', 0.0):
            unseen = 0.0                                  # ball just placed on the centre spot by the referee
        if unseen > self.get_parameter('rl_search_after').value and self._role() not in ('defend', 'wait', 'regroup', 'regroup_def'):
            # (a defender that lost the ball keeps its goal - no search walk; see the role code below)
            # safety net for the game flow: the policy does not find the ball -> first turn to where it was last
            # seen (quick), then the proven search routine (back up, spin, roam)
            self.cancel_nav()
            self.play_ctx = (pose, self.estimate_opponent(pose), None)
            look = self.get_parameter('rl_search_after').value + 2.0
            err = wrap(math.atan2(self.ball[1] - pose[1], self.ball[0] - pose[0]) - pose[2]) \
                if self.ball is not None else 0.0
            near = self.ball is not None and math.hypot(self.ball[0] - pose[0], self.ball[1] - pose[1]) < 0.35
            if near and unseen < look + 2.0:
                # the ball should be right in front but the camera does not see it (too close, under the image):
                # face the remembered spot and back up a little so it comes into view
                cmd = Twist()
                cmd.angular.z = clip(2.0 * err, -0.8, 0.8)
                from pinky_soccer_game.rl_policy import over_line_ahead
                back = -0.12
                if over_line_ahead(pose, back, self.L / 2, self.W / 2, self.G) or \
                        goal_mouth_guard(pose, back, self.L / 2, self.G) == 0.0:
                    # line or a goal behind (keeper in its mouth, r9s2): creep to the remembered spot instead,
                    # a touch moves the ball
                    back = 0.08
                cmd.linear.x = back if abs(err) < 0.5 else 0.0
                self.cmd_pub.publish(cmd)
                self.set_state('NEAR_BACKUP')
                self.search_phase = None
            elif self.ball is not None and unseen < look and abs(err) > 0.25:
                cmd = Twist()
                cmd.angular.z = clip(2.5 * err, -1.6, 1.6)   # ball out of view: turn fast to look (user)
                self.cmd_pub.publish(cmd)
                self.set_state('LOOK_TOWARD')
                self.search_phase = None
            else:
                self.search(pose[0], pose[1], pose[2], now)
            self.rl_t = 0.0                                  # fresh policy step as soon as the ball is back
            return
        self.search_phase = None
        if self.rl is not None and now - self.rl_t >= 0.095:
            seen = self.ball is not None and now - self.ball_t < 0.25
            since = (now - self.ball_t) if self.ball is not None else 5.0
            opp = self.estimate_opponent(pose)
            self.rl.act(pose, self.ball, seen, since, opp)    # keeps the policy's memory (ball speed) current
            self.rl_t = now
        # ball out of the field (over a line, waiting for the referee's 'ball to centre'): no chasing it into a
        # goal or over the line - stand still facing it (blue kept walking into its own goal for it, final5f)
        b = self.ball
        if b is not None and (abs(b[0]) > self.L / 2 + 0.01 or abs(b[1]) > self.W / 2 + 0.01) and \
                self._role() in ('defend', 'attack'):
            # attack / defence game, ball out: the roles have just swapped - the new defender runs back to its
            # goal at once, the new attacker lines up behind the centre spot for the ball coming back there
            from pinky_soccer_game.rl_policy import defend_cmd
            vmax = self.get_parameter('v_dribble').value
            if self._role() == 'defend':
                from pinky_soccer_game import ball_play
                v, w, mode = ball_play.defend_cmd(pose, None, 9.0, -self.goal[0], self.L / 2, self.W / 2, self.G,
                                                  self.rl_pp, now)
                tag = 'DEFEND_' + mode
            else:
                gx = self.goal[0]
                tx, ty = -math.copysign(0.25, gx), 0.0              # 0.25 m behind the centre spot
                d = math.hypot(tx - pose[0], ty - pose[1])
                if d > 0.05:
                    err = wrap(math.atan2(ty - pose[1], tx - pose[0]) - pose[2])
                    v = 0.0 if abs(err) > 0.5 else vmax * max(0.3, math.cos(err))
                    w = clip(2.0 * err, -0.8, 0.8)
                else:
                    err = wrap(math.atan2(0.0 - pose[1], 0.0 - pose[0]) - pose[2])
                    v, w = 0.0, (clip(2.0 * err, -0.8, 0.8) if abs(err) > 0.1 else 0.0)
                tag = 'ATTACK_READY'
            cmd = Twist()
            cmd.linear.x = goal_mouth_guard(pose, v, self.L / 2, self.G, nose=0.09 + 0.25 * abs(v))
            cmd.angular.z = w
            self.cancel_nav()
            self.cmd_pub.publish(cmd)
            self.rl_pp['mode'] = 'WAIT'
            self.set_state(tag)
            return
        if b is not None and (abs(b[0]) > self.L / 2 + 0.01 or abs(b[1]) > self.W / 2 + 0.01):
            err = wrap(math.atan2(b[1] - pose[1], b[0] - pose[0]) - pose[2])
            cmd = Twist()
            cmd.angular.z = clip(2.0 * err, -0.8, 0.8) if abs(err) > 0.2 else 0.0
            self.cancel_nav()
            self.cmd_pub.publish(cmd)
            self.rl_pp['mode'] = 'WAIT'
            self.set_state('BALL_OUT_WAIT')
            return
        role = self._role()
        if role == 'regroup':
            # new attacker after a time-up swap: to the spot in our own half (start pose, or just behind the
            # ball when it lies there) at full speed, then face the ball and wait for the referee
            from pinky_soccer_game.ball_play import regroup_spot
            own_x = -self.goal[0]
            sx, sy = regroup_spot(own_x, b, self.L / 2, self.W / 2, self.get_parameter('ready_offset').value)
            face = math.atan2(b[1] - sy, b[0] - sx) if b is not None else self.attack_yaw
            self.cancel_nav()
            self.play_ctx = (pose, self.estimate_opponent(pose), None)
            self.publish_play(self.direct_goto(pose[0], pose[1], pose[2], np.array([sx, sy]), face, pos_tol=0.05,
                                               vmax=self.get_parameter('v_dribble').value))
            self.rl_pp.clear()
            self.rl_pp['mode'] = 'WAIT'
            self.set_state('REGROUP')
            return
        if role == 'regroup_def':
            # the new defender after a swap: straight to its goal, fast, facing the play (the referee says when
            # play goes on)
            own_x = -self.goal[0]
            inward = -math.copysign(1.0, own_x)
            sx, sy = own_x + inward * 0.35, 0.0
            face = math.atan2(b[1] - sy, b[0] - sx) if b is not None else -self.attack_yaw
            self.cancel_nav()
            self.play_ctx = (pose, self.estimate_opponent(pose), None)
            self.publish_play(self.direct_goto(pose[0], pose[1], pose[2], np.array([sx, sy]), face, pos_tol=0.05,
                                               vmax=self.get_parameter('v_dribble').value))
            self.rl_pp.clear()
            self.rl_pp['mode'] = 'WAIT'
            self.set_state('REGROUP_DEF')
            return
        if role == 'wait':
            # the other team kicks off: stay where we are (outside the centre circle), face the ball
            err = wrap(math.atan2(b[1] - pose[1], b[0] - pose[0]) - pose[2]) if b is not None else 0.0
            cmd = Twist()
            cmd.angular.z = clip(2.0 * err, -0.8, 0.8) if abs(err) > 0.2 else 0.0
            self.cancel_nav()
            self.cmd_pub.publish(cmd)
            self.rl_pp['mode'] = 'WAIT'
            self.set_state('KICKOFF_WAIT')
            return
        if role == 'defend':
            from pinky_soccer_game.rl_policy import defend_cmd
            own_x = -self.goal[0]
            from pinky_soccer_game.rl_policy import ball_filter
            bf = ball_filter(self.rl_pp, b) if b is not None else None
            from pinky_soccer_game import ball_play
            age = 0.0 if getattr(self, 'carrying', False) else now - self.ball_t
            v, w, mode = ball_play.defend_cmd(pose, bf, age, own_x, self.L / 2, self.W / 2, self.G,
                                              self.rl_pp, now, give_way=self._give_way(pose),
                                              opp=self.estimate_opponent(pose))
            self.rl_pp['mode'] = 'WAIT' if mode in ('KEEP', 'LOOK', 'BLOCK_HOLD', 'WANDER_LOOK') else mode
            if mode.startswith(('CLEAR_PUSH', 'CLEAR_SHOT', 'CLEAR_KICK', 'CLEAR_WINDUP', 'CLEAR_TAP')):
                self.push_t = time.time()
            self.rl_cmd = (v, w)
            # the defender moves at half the attacker's speed - whatever it does (user, 2026-11-26)
            vd = ball_play.P['v_def']                      # half the attacker's play speed
            if mode in ('ROAM', 'RUN_BACK'):
                vd = ball_play.P['roam_v']                # roaming / running back to the goal: a little faster
            v = max(-vd, min(vd, v))
            cmd = Twist()
            cmd.linear.x = goal_mouth_guard(pose, v, self.L / 2, self.G, nose=0.09 + 0.25 * abs(v))
            cmd.angular.z = w
            self.cancel_nav()
            self.cmd_pub.publish(cmd)
            self.set_state('DEFEND_' + mode)
            return
        # approach and dribble in straight lines (the learned policy weaved: 1.8 turn reversals/s; straight
        # control in the same test: 0.3/s and twice the goals) - every tick, 20 Hz
        from pinky_soccer_game.rl_policy import straight_cmd
        aim = self.goal
        if b is not None and abs(b[0] - self.goal[0]) < 0.15 and abs(b[1]) > self.G / 2 - 0.03:
            # ball on the end line beside the goal: a push at the goal centre runs into the line (line guard
            # stops it every time, final5i) - first along the line into the mouth, then the normal shot
            aim = np.array([self.goal[0] - math.copysign(0.12, self.goal[0]), 0.0])
        from pinky_soccer_game.rl_policy import ball_filter
        bf = ball_filter(self.rl_pp, self.ball) if self.ball is not None else None
        self.rl_pp['ball_age'] = now - self.ball_t           # the orbit looks at the ball when it is > 1 s
        # look - small move - look again (rule-based, 2026-11-26: the learned policy is gone)
        from pinky_soccer_game import ball_play
        age = 0.0 if getattr(self, 'carrying', False) else now - self.ball_t
        opp_now = self.estimate_opponent(pose)
        v, w, mode = ball_play.attack_cmd(pose, bf, age, aim, self.L / 2, self.W / 2, self.rl_pp, now, goal_w=self.G,
                                          opp=opp_now,
                                          free=self._ball_free(pose),
                                          # roles set by the referee: the attacker never gives way to the
                                          # keeper (the robot guard keeps them apart) - j01 yielded the ball
                                          # the robot closer to the ball touches it first, the other one
                                          # steps back (user, 2026-11-26) - in the role game too
                                          give_way=self._give_way(pose) if self._role() is None else False)
        # stand-off with the keeper (user, 2026-11-26: yellow brought the ball up to blue and both stood still -
        # blue should push it away): the ball at our nose, the keeper close to it and the ball not moving for
        # 1.5 s -> step back ~18 cm; the ball is then loose and the keeper clears it
        if self._role() == 'attack' and bf is not None and opp_now is not None:
            near = math.hypot(bf[0] - pose[0], bf[1] - pose[1]) < 0.20 and \
                math.hypot(bf[0] - opp_now[0], bf[1] - opp_now[1]) < 0.40
            ref = getattr(self, 'standoff_ref', None)
            if not near or ref is None or math.hypot(bf[0] - ref[0], bf[1] - ref[1]) > 0.02:
                self.standoff_ref = (bf[0], bf[1], now)
            elif now - ref[2] > 1.5:
                self.standoff_until, self.standoff_ref = now + 2.2, None   # ~18 cm at the halved speed
                self.get_logger().info('stand-off with the keeper: stepping back, the ball is its')
        if now < getattr(self, 'standoff_until', 0.0):
            err = wrap(math.atan2(bf[1] - pose[1], bf[0] - pose[0]) - pose[2]) if bf is not None else 0.0
            v, w, mode = -0.08, clip(2.0 * err, -0.8, 0.8), 'STANDOFF_BACK'
        self.rl_pp['mode'] = mode
        if mode in ('PUSH', 'SHOT', 'SHOT_BACK', 'KICK', 'WINDUP', 'TAP', 'TAP_FLICK', 'TAP_DODGE', 'LINE_TAP', 'LINE_FLICK'):
            self.push_t = time.time()
        self.rl_cmd = (v, w)
        self.cancel_nav()
        cmd = Twist()
        if now < self.rl_escape_until:
            # safety net: the line guard kept blocking the policy -> face the field centre for a moment
            err = wrap(math.atan2(-pose[1], -pose[0]) - pose[2])
            cmd.angular.z = clip(2.0 * err, -0.8, 0.8)
            cmd.linear.x = 0.10 if abs(err) < 0.4 else 0.0
            self.cmd_pub.publish(cmd)
            self.set_state('RL_ESCAPE')
            return
        cmd.linear.x, cmd.angular.z = self.rl_cmd
        # the robot's nose never goes under a crossbar (it got stuck in the goal; the ball still goes in)
        cmd.linear.x = goal_mouth_guard(pose, cmd.linear.x, self.L / 2, self.G,
                                        nose=0.09 + 0.25 * abs(cmd.linear.x))   # + braking at full speed
        self.guard_hit = False
        self.cmd_pub.publish(cmd)                    # line guard still applies
        if self.guard_hit:
            self.guard_since = self.guard_since or now
            if now - self.guard_since > 1.5:
                self.rl_escape_until, self.guard_since = now + 1.0, None
                self.get_logger().info('RL policy blocked by the boundary line for 1.5 s -> turn to the centre')
        else:
            self.guard_since = None
        self.set_state('PLAY_' + self.rl_pp.get('mode', 'BEHIND'))
        # diagnostics: going to the ball but hardly moving (g14: 11 s ROUND, 15 cm) -> say what holds it back
        ref = getattr(self, 'slow_ref', None)
        py = getattr(self, 'slow_yaw', None)
        self.slow_turn = getattr(self, 'slow_turn', 0.0) + (abs(wrap(pose[2] - py)) if py is not None else 0.0)
        self.slow_yaw = pose[2]
        m = self.rl_pp.get('mode')
        if m != getattr(self, 'slow_mode', None):
            self.slow_flips = getattr(self, 'slow_flips', 0) + 1
            self.slow_mode = m
        if ref is None or now - ref[2] > 3.0:
            if ref is not None and math.hypot(pose[0] - ref[0], pose[1] - ref[1]) < 0.06 and \
                    self.rl_pp.get('mode') in ('BEHIND', 'ROUND', 'PUSH', 'APPROACH', 'ORBIT_STEP', 'ORBIT_LOOK'):
                cin = getattr(self.cmd_pub, 'last_in', None)
                cout = getattr(self.cmd_pub, 'last_out', None)
                opp = self.opp if (self.opp is not None and now - self.opp_t < 2.0) else None
                self.get_logger().info(
                    f'slow {self.rl_pp.get("mode")}: pose ({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}) '
                    f'ball {None if self.ball is None else tuple(round(float(q), 2) for q in self.ball)} '
                    f'seen {now - self.ball_t:.1f}s ago, opp {None if opp is None else tuple(round(float(q), 2) for q in opp[:2])}, '
                    f'cmd {self.rl_cmd[0]:+.2f},{self.rl_cmd[1]:+.2f} -> mouth {cin} -> guards {cout}; '
                    f'turned {self.slow_turn:.1f} rad, {self.slow_flips} mode changes in 3 s')
            self.slow_ref = (pose[0], pose[1], now)
            self.slow_turn, self.slow_flips = 0.0, 0

    def _line(self, m):
        self.line_pts = np.array([[q.position.x, q.position.y] for q in m.poses]).reshape(-1, 2)
        self.line_t = time.time()

    def _robot_guard(self, cmd):
        """No collisions: the other robot (lidar estimate, fresh) in the way -> slow down / stop that motion."""
        if not self.get_parameter('robot_guard').value or self.opp is None:
            return cmd
        pose = self.robot_pose()
        if pose is None or time.time() - self.opp_t > self._opp_hold(pose):
            return cmd
        rx, ry, ryaw = pose
        dx, dy = self.opp[0] - rx, self.opp[1] - ry
        c, s = math.cos(ryaw), math.sin(ryaw)
        from pinky_soccer_game.rl_policy import has_priority, robot_guard_vw
        # right of way: the robot closer to the ball (camera ball, lidar opponent) goes first; tie -> yellow
        ball = self.ball if (self.ball is not None and time.time() - self.ball_t < 2.0) else None
        pushing = time.time() - getattr(self, 'push_t', 0.0) < 1.5
        if ball is None and pushing and getattr(self, 'had_priority', None) is not None:
            prio = True       # dribbling: the ball under the nose drops out of the camera - still ours (yellow
                              # kept giving way while dribbling, final5g)
        elif ball is None:
            prio = False                                 # cannot see the ball: never claim the right of way
        else:                                            # (it held its ground blind in front of its goal, r4s1)
            # strictly by the distance to the ball (user, 2026-11-26: no collisions - whoever is closer touches
            # first); the old 'ball at our nose -> right of way' let BOTH robots claim it over a ball between
            # them and drive into each other (p02).  A near tie: the attacker (free game: yellow) goes first.
            role = self._role()
            tie_mine = role == 'attack' or role == 'kickoff' or (role is None and
                                                                 self.get_parameter('team').value == 'yellow')
            own_x = -self.goal[0]
            if role in ('attack', 'kickoff'):
                prio = True       # the attacker has the right of way in its attack (p05 C) ...
            elif role == 'defend' and str(self.rl_pp.get('mode', '')).startswith('WANDER'):
                prio = False      # the wandering keeper always gives way (it is only an obstacle)
            elif role == 'defend' and 'DANGER' in str(self.rl_pp.get('mode', '')):
                prio = True       # ball right in front of our goal: the keeper goes for it (user, 2026-11-26)
            elif role == 'defend' and not str(self.rl_pp.get('mode', '')).startswith('CLEAR'):
                prio = False      # ... the defender always gives way at the ball (user, 2026-11-26: it only
                                  # blocks the shot) - unless it is clearing a loose ball at its goal
            else:
                prio = has_priority((rx, ry), self.opp, ball, tie_mine, had=getattr(self, 'had_priority', None))
        had = getattr(self, 'had_priority', None)
        if had is not None and prio != had and time.time() - getattr(self, 'prio_t', 0.0) < 1.0:
            prio = had                                   # keep a right-of-way decision at least 1 s
        if prio != had:
            self.had_priority, self.prio_t = prio, time.time()
            if ball is not None:
                self.get_logger().info('right of way at the ball: ' + ('mine - going in' if prio else
                                                                     'the other robot is closer - giving way'))
        v, w = robot_guard_vw(cmd.linear.x, cmd.angular.z, (c * dx + s * dy, -s * dx + c * dy), prio)
        if v != cmd.linear.x:
            self.robot_stops = getattr(self, 'robot_stops', 0) + 1
            if v == 0.0 and self.robot_stops % 20 == 1:
                self.get_logger().info(f'other robot {math.hypot(dx, dy):.2f} m in the way: stop')
        v = self._pinch_yield(v, (rx, ry), (c * dx + s * dy), math.hypot(dx, dy), ball, prio)
        cmd.linear.x, cmd.angular.z = v, w
        return cmd

    def _ball_free(self, pose):
        """No other robot within 0.5 m of the ball: room for a shot (step back + hit)."""
        opp = self.estimate_opponent(pose)
        return opp is None or self.ball is None or math.hypot(self.ball[0] - opp[0], self.ball[1] - opp[1]) > 0.5

    def _give_way(self, pose):
        """Face-off: the other robot is close to the ball (both within 0.45 m) and has the right of way (closer,
        same rule as the collision guard) -> we step back and let it play."""
        opp = self.estimate_opponent(pose)
        if opp is None or self.ball is None or time.time() - self.ball_t > 1.0:
            return False
        b = self.ball
        d_me, d_opp = math.hypot(b[0] - pose[0], b[1] - pose[1]), math.hypot(b[0] - opp[0], b[1] - opp[1])
        if d_me > 0.45 or d_opp > 0.45 or d_me < 0.20 or getattr(self, 'had_priority', None) is not False:
            return False                     # (ball at our own nose: never step back - play it boldly)
        # both near the ball and the other robot is closer: it touches first, we step back (user, 2026-11-26:
        # whoever is closer to the ball goes - no collisions)
        return True

    def _kick_circle_guard(self, cmd):
        """Kick-off: the waiting robot does not drive into the 0.30 m circle round the centre spot until the
        ball has left it (or 8 s passed) - turning is free."""
        if not getattr(self, 'kick_circle', False) or self.match_phase != 'PLAY':
            return cmd
        b = self.ball if (self.ball is not None and time.time() - self.ball_t < 1.0) else None
        if (b is not None and math.hypot(b[0], b[1]) > 0.08) or time.time() - self.kick_circle_t > 8.0:
            self.kick_circle = False
            return cmd
        pose = self.robot_pose()
        if pose is None or abs(cmd.linear.x) < 1e-3:
            return cmd
        x, y, yaw = pose
        d = math.hypot(x, y)
        inward = (math.cos(yaw) * -x + math.sin(yaw) * -y) * cmd.linear.x > 0    # this motion gets closer
        if d < 0.34 and inward:
            cmd.linear.x = 0.0
        return cmd

    def _pinch_yield(self, v, me, opp_ahead, d_opp, ball, prio):
        """Ball squeezed between the two robots and not moving (r6s2: 10 s standstills): the robot without the
        right of way backs away until the robots are 0.34 m apart; still stuck after 5 s -> blue always yields
        (both may claim the right of way for a moment)."""
        now = time.time()
        if ball is None or d_opp > 0.34 or math.hypot(ball[0] - me[0], ball[1] - me[1]) > 0.18:
            self.pinch_ref = None
            return v
        ref = getattr(self, 'pinch_ref', None)
        if ref is None or math.hypot(ball[0] - ref[1][0], ball[1] - ref[1][1]) > 0.04:   # camera jitter close up
            self.pinch_ref = (now, np.array(ball))
            return v
        still = now - ref[0]
        if (not prio and still > 2.0) or (still > 5.0 and self.get_parameter('team').value == 'blue'):
            if getattr(self, 'pinch_log_t', 0.0) < now - 3.0:
                self.pinch_log_t = now
                self.get_logger().info('ball squeezed between the robots: backing away')
            return -0.10 if opp_ahead > 0 else 0.10
        return v

    def _line_guard(self, cmd):
        """Boundary points (base frame, from the camera) inside the strip the robot would sweep next -> no
        motion in that direction (turning stays allowed, so the robot can face away and carry on).
        First the same check from the robot's own map pose (any direction - the camera only sees ahead)."""
        if not self.get_parameter('line_guard').value:
            return cmd
        pose = self.robot_pose()
        if pose is not None and abs(cmd.linear.x) > 1e-3:
            from pinky_soccer_game.rl_policy import over_line_ahead
            if over_line_ahead(pose, cmd.linear.x, self.L / 2, self.W / 2, self.G):
                cmd.linear.x = 0.0
                self.guard_hit = True                        # the RL escape logic counts this too
                self.line_stops += 1
                if self.line_stops % 20 == 1:
                    self.get_logger().info('boundary line (map) in the way: stop')
                return cmd
        if time.time() - self.line_t > 1.0 or not len(self.line_pts):
            return cmd
        v = cmd.linear.x
        if abs(v) < 1e-3:
            return cmd
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        p = self.line_pts
        lateral = np.abs(p[:, 1]) < g('robot_half_width') + 0.01
        reach = g('line_margin') + 0.25 * abs(v)                    # + braking distance (smoother, 0.25 s)
        if v > 0:
            hit = lateral & (p[:, 0] > -0.02) & (p[:, 0] < g('robot_front') + reach)
        else:
            hit = lateral & (p[:, 0] < 0.02) & (p[:, 0] > -(g('robot_back') + reach))
        self.guard_hit = bool(hit.any())
        if hit.any():
            cmd.linear.x = 0.0
            self.line_stops += 1
            if self.line_stops % 20 == 1:
                self.get_logger().info(f'boundary line {"ahead" if v > 0 else "behind"}: stop '
                                       f'({int(hit.sum())} line points)')
        return cmd

    def concede_point(self, pose):
        """Centre-line point on our own side, 0.15 m from the line, to the side away from the other robot."""
        n = np.array([math.cos(self.attack_yaw), math.sin(self.attack_yaw)])
        opp = self.estimate_opponent(pose) if self.get_parameter('use_opponent').value else None
        side = -1.0 if (opp is not None and opp[1] > 0) else 1.0
        return -n * 0.15 + np.array([0.0, side * self.W / 4])

    # ------------------------------------------------------------------ helpers
    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', self.get_parameter('base_frame').value,
                                                rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        pose = (t.transform.translation.x, t.transform.translation.y, yaw_from_quat(t.transform.rotation))
        self._localisation_guard(pose)
        return pose

    def _localisation_guard(self, pose):
        """Wheel slip (pushing the other robot / a wall) can carry AMCL off; with its particles packed tight it
        does not come back by itself.  The robot centre can never be within 6 cm of a wall: if AMCL says so,
        re-seed it around the nearest possible point with a wide spread (0.3 m, same heading - the field is
        point-symmetric, a heading reset could flip it) so the lidar pulls it back onto the walls."""
        x, y, yaw = pose
        strip = self.get_parameter('outer_strip').value
        hx, hy = self.L / 2 + strip - 0.06, self.W / 2 + strip - 0.06
        if (abs(x) <= hx + 0.03 and abs(y) <= hy + 0.03) or time.time() - self.reseed_t < 3.0:
            return
        self.reseed_t = time.time()
        cx, cy = clip(x, -hx + 0.05, hx - 0.05), clip(y, -hy + 0.05, hy - 0.05)
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.pose.pose.position.x, m.pose.pose.position.y = cx, cy
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.09
        m.pose.covariance[35] = 0.03
        self.initpose_pub.publish(m)
        self.get_logger().warning(f'localisation outside the field ({x:+.2f}, {y:+.2f}) -> AMCL re-seeded around '
                                  f'({cx:+.2f}, {cy:+.2f})')

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

    def _opp_hold(self, pose):
        """How long the last lidar opponent stays valid: 0.6 s, but 2 s when it was close - right next to us its
        body is inside the lidar's minimum range (< 0.10 m), so it vanishes exactly when the guard needs it."""
        if self.opp is not None and pose is not None and \
                math.hypot(self.opp[0] - pose[0], self.opp[1] - pose[1]) < 0.30:
            return 2.0
        return 0.6

    def estimate_opponent(self, pose):
        """Lidar returns that are not explained by the static map -> opponent robot (nearest cluster)."""
        if not self.get_parameter('use_opponent').value:
            return None
        if self.scan is None or self.map is None or pose is None:
            return self.opp if time.time() - self.opp_t < self._opp_hold(pose) else None
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
        if self.opp is not None and time.time() - self.opp_t < self._opp_hold(pose):
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
            pose = self.robot_pose()
            if self.get_parameter('concede_centre').value and pose is not None:
                self.concede_xy = self.concede_point(pose)
                self.concede_t0, self.concede_hold_t = time.time(), None
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
            yaw_err = abs(wrap(ryaw - ready_yaw))
            if self.match_phase == 'READY':
                # hysteresis: the localisation jitters by a few cm - stay READY (and still) unless the robot
                # really got moved off its start pose (then line up again)
                if dist < 0.12 and yaw_err < 0.35:
                    self.cancel_nav()
                    self.stop()
                    self.set_state('READY')
                    return
                self.match_phase = 'PREPARE'
                self.router.reset()
            if self.concede_xy is not None:
                # conceded: first to the centre-line point (out of the scorer's way), wait, then back
                cdist = float(np.linalg.norm(np.array([rx, ry]) - self.concede_xy))
                if cdist < 0.08 or time.time() - self.concede_t0 > 20.0:
                    if self.concede_hold_t is None:
                        self.concede_hold_t = time.time()
                    if time.time() - self.concede_hold_t < self.get_parameter('concede_hold').value:
                        self.cancel_nav()
                        self.stop()
                        self.set_state('CONCEDE_CENTRE')
                        return
                    self.concede_xy = None
                    self.router.reset()
                    self.cancel_nav()
                else:
                    ready_pose, ready_yaw = self.concede_xy, self.attack_yaw
                    dist, yaw_err = cdist, 1.0                # never READY on the way to the centre line
            if dist < 0.05 and yaw_err < 0.15:
                self.cancel_nav()
                self.stop()
                self.match_phase = 'READY'
                self.set_state('READY')
            else:
                opp = self.estimate_opponent(pose)
                if time.time() - getattr(self, 'prepare_t0', time.time()) > 12.0:
                    opp = None                       # detours did not get us there: straight to the start pose
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
                self.set_state('CONCEDE_GO' if self.concede_xy is not None else 'PREPARING')
            return

        me_team = self.get_parameter('team').value.upper()
        if self.match_control and time.time() < self.hold_until:
            # we scored the last goal: stand still for scorer_delay after the restart (fair chances)
            self.cancel_nav()
            self.stop()
            self.set_state('SCORER_WAIT')
            return
        if self.rl is not None or self.get_parameter('ball_play').value == 'rules':
            now_t = time.time()
            # safety net for the game flow: the policy standing still in play (r3s3: 10 s next to the ball)
            ref = getattr(self, 'rl_idle_ref', None)
            if ref is None or math.hypot(rx - ref[0], ry - ref[1]) > 0.03 or abs(wrap(ryaw - ref[2])) > 0.2:
                self.rl_idle_ref = (rx, ry, ryaw, now_t)
            elif self.rl_pp.get('mode') in ('WAIT', 'GIVE_WAY', 'AIM'):
                self.rl_idle_ref = (rx, ry, ryaw, now_t)    # waiting / giving way on purpose: not idle
            elif self._role() is not None:
                pass        # attack / defence game: the unjam step in _rl_play frees it, the old rule-based
                            # states (ORBIT / BACKOFF) only backed into the line and froze for minutes (g06)
            elif now_t - ref[3] > self.get_parameter('rl_idle_after').value and now_t >= getattr(self, 'rl_rules_until', 0):
                self.rl_rules_until = now_t + self.get_parameter('rl_rules_for').value
                self.rl_idle_ref = None
                self.get_logger().info('RL policy idle -> rule-based play for a moment')
            if now_t >= getattr(self, 'rl_rules_until', 0.0):
                self._rl_play((rx, ry, ryaw), now_t)
                return
            # else: fall through to the rule-based state machine below
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
        if now < self.escape_until:
            # jammed again and again (e.g. hooked on a goal post): after the long back-off turn to the
            # field centre and drive a little towards it before playing on
            self.cancel_nav()
            err = wrap(math.atan2(-ry, -rx) - ryaw)
            cmd = Twist()
            cmd.angular.z = clip(2.5 * err, -2.0, 2.0)
            cmd.linear.x = 0.10 if abs(err) < 0.5 else 0.0
            self.cmd_pub.publish(cmd)
            self.set_state('ESCAPE')
            return
        ref = self.stuck_ref
        if ref is None or math.hypot(rx - ref[0], ry - ref[1]) > 0.02 or abs(wrap(ryaw - ref[2])) > 0.1 \
                or self.state in ('NO_ROOM', 'KICKOFF_WAIT', 'OUT'):    # standing still on purpose
            self.stuck_ref = (rx, ry, ryaw, now)
        elif now - ref[3] > self.get_parameter('stuck_time').value:
            near = self.nearest_obstacle()
            ball_far = self.ball is None or float(np.linalg.norm(self.ball - r)) > 0.12
            if (near is not None and near[0] < self.get_parameter('jam_dist').value) or ball_far:
                self.jam_count = self.jam_count + 1 if now - self.jam_last_t < 8.0 else 1
                self.jam_last_t = now
                dist = self.get_parameter('backoff_dist').value
                if self.jam_count >= 2:                      # the short back-off did not free us
                    dist = max(dist, 0.15)
                speed = 0.10
                front = near is None or near[1]
                self.backoff_v = -speed if front else speed  # away from whatever we are pressed against
                self.backoff_until = now + dist / speed
                self.escape_until = self.backoff_until + (1.5 if self.jam_count >= 2 else 0.0)
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
        if abs(b[0]) > self.L / 2 + 0.04 and abs(b[1]) > self.G / 2:       # 4 cm: camera estimate error
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

        # ---- APPROACH: behind the ball, facing the aim point (state set below: AVOID when detouring)
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
            now = time.time()
            hold = getattr(self, 'detour_hold', None)
            if side_step:                                # keep a chosen detour >= 1 s (it flipped at 20 Hz
                self.detour_hold = (now + 1.0, tgt)      # APPROACH <-> AVOID right next to the other robot, r8s1)
            elif hold is not None and now < hold[0]:
                tgt, side_step = hold[1], True
            if side_step:
                self.set_state('AVOID')
                self.publish_play(self.direct_goto(rx, ry, ryaw, np.array(tgt), line_ang, pos_tol=0.0,
                                                   vmax=self.get_parameter('v_approach').value))
                return
            self.set_state('APPROACH')
            self.publish_play(self.direct_goto(rx, ry, ryaw, behind_pt, line_ang, pos_tol=0.03,
                                                  vmax=self.get_parameter('v_approach').value))
            return
        self.set_state('APPROACH')
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
        cmd.linear.x = goal_mouth_guard(me, cmd.linear.x, self.L / 2, self.G,
                                        nose=0.09 + 0.25 * abs(cmd.linear.x))   # + braking at full speed
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
