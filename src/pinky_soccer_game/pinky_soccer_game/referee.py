#!/usr/bin/env python3
"""Referee: detects goals from Gazebo ground truth, keeps score, re-spots the ball.

Subscribes  /gz/dynamic_pose
Publishes   /game/score  (std_msgs/String  "YELLOW 2 - 1 BLUE")   /game/event (String)
            /game/kickoff (String, 10 Hz)  "YELLOW"/"BLUE" = that team kicks off and must touch the ball
                                           first, "OPEN" = ball in play for both
            /game/phase   (String, 5 Hz)  WAIT | PREPARE | START | GOAL | FINISHED   (match_control only)
            /game/goals   (String, 5 Hz)  goal log, one goal per line "mm:ss|TEAM|yellow|blue|kind"
                                          (kind: 선제골 / 추가골 / 추격골 / 동점골 / 역전골)
            /game/control PREPARE / START / GOAL / WAIT  -> the robots (the referee runs the match itself)
Match flow (match_control:=true, buttons in the match UI send NEW_GAME / RESET / UNDO_GOAL on /game/control):
            NEW_GAME  -> 0:0, PREPARE: both robots drive to their start poses, ball to the centre
            both robots at their start poses -> ball placed for the kickoff -> START after `start_delay`
            goal      -> counted automatically, robots stop, after `goal_pause` PREPARE again (centre kickoff)
            `win_score` reached (default 5) -> FINISHED, robots stop
            RESET     -> something went wrong: back to the start poses and redo the last kickoff (score kept)
            UNDO_GOAL -> wrongly detected goal: take the last goal back and redo the kickoff before it
Kickoff:    (default, match rule) none: every start / restart puts the ball on the centre spot, both go at once.
            kickoff_rule:=true: the first kickoff is YELLOW, after a goal the team that conceded kicks off.
            Match start: ball `kickoff_gap` in front of the kicking robot's bumper; after a goal: centre spot
            (match rule).  After START the other team waits until the ball has moved `kickoff_touch`
            (or `kickoff_timeout` passed).
Out:        behind a goal line beside the goal -> back on the centre spot (match rule), play on.
Corner:     ball caught in a corner (`corner_zone` from both walls) and not moving for `wedge_timeout`
            -> back on the centre spot (match rule; no random spots, no resets in the open field).
            A ball lying alone is never moved
            (`stuck_timeout` > 0 would bring back the old 'not moving at all' reset).
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from pinky_soccer_sim.gz_utils import PoseTracker, gz_set_pose
from pinky_soccer_sim.field import default_dims

# defaults = the default field (pinky_soccer_sim/config/field_hardboard.yaml); launch files pass the real values
FIELD_L, FIELD_W, FIELD_G, FIELD_STRIP = default_dims()


class Referee(Node):
    def __init__(self):
        super().__init__('referee')
        self.declare_parameter('field_length', FIELD_L)
        self.declare_parameter('field_width', FIELD_W)
        self.declare_parameter('goal_width', FIELD_G)
        self.declare_parameter('world', 'soccer')
        self.declare_parameter('celebrate_timeout', 60.0)    # s max for the scorer's celebration (drive + 3 turns)
        self.declare_parameter('sim_backend', 'gz')          # gz | mujoco (how the ball / a robot is moved)
        self.declare_parameter('yellow_initialpose_topic', '/initialpose')   # duel: /yellow/initialpose
        self.declare_parameter('ball_radius', 0.02135)
        self.declare_parameter('stuck_timeout', 10.0)      # s without ANY ball movement (deadlock: both robots
                                                           # stuck next to it) -> centre; 0 = off
                                                           # (match rule: a ball lying alone is left alone)
        self.declare_parameter('wedge_timeout', 5.0)       # s, ball stuck together with a robot -> centre
        self.declare_parameter('wall_zone', 0.06)          # m: ball this close to the boundary LINE (or beyond
                                                           # it; the robots must not cross the line) ...
        self.declare_parameter('wall_stuck_time', 3.0)     # ... and not moving this long -> centre (robots
                                                           # cannot get behind a ball lying on the wall)
        self.declare_parameter('corner_zone', 0.20)        # m from both walls = in a corner
        self.declare_parameter('wedge_wall_dist', 0.15)    # ball centre this close to a wall = 'at the wall'
        self.declare_parameter('wedge_robot_dist', 0.13)   # robot centre this close to the ball = pressed on it
        self.declare_parameter('outer_strip', FIELD_STRIP)  # playing area -> wall distance
        self.declare_parameter('out_timeout', 0.5)         # s behind a goal line beside the goal -> centre
        self.declare_parameter('ready_offset', 0.38)       # start pose: own goal line + this (striker/opponent)
        self.declare_parameter('robot_front', 0.049)       # bumper front, ahead of base_footprint
        # match rule: no kickoff team - every (re)start puts the ball on the centre spot and both robots go at once.
        # kickoff_rule:=true brings back the old kickoff (first yellow, then the conceding team, 5 cm ball)
        self.declare_parameter('kickoff_rule', False)
        self.declare_parameter('kickoff_gap', 0.05)        # ball gap in front of the kicking robot's bumper
        self.declare_parameter('kickoff_touch', 0.03)      # ball moved this far -> kickoff taken
        self.declare_parameter('kickoff_timeout', 2.0)     # s; then the ball is open for both anyway
        self.declare_parameter('reset_robots', False)      # teleporting the robot breaks AMCL; keep False
        self.declare_parameter('match_seconds', 0.0)       # 0 = unlimited
        self.declare_parameter('match_control', False)
        self.declare_parameter('win_score', 5)             # first to this many goals wins (0 = endless)
        self.declare_parameter('goal_pause', 2.0)          # s robots stand still after a goal (celebration)
        self.declare_parameter('start_delay', 1.5)         # s after the kickoff ball is placed -> START
        self.declare_parameter('ready_pos_tol', 0.13)      # start pose tolerance (ground truth)
        self.declare_parameter('ready_yaw_tol', 0.30)
        self.declare_parameter('require_robot_ready', True)   # like the real referee: the strikers must
                                                               # also REPORT READY (their own localisation)
        self.declare_parameter('prepare_timeout', 40.0)    # s; start anyway if a robot cannot get ready
        self.declare_parameter('jam_timeout', 3.0)         # s a robot sits inside a goal mouth -> lifted out
        # match UI: '경기 준비' puts the ball on the centre spot (in the real match a person does this)
        self.declare_parameter('respot_on_prepare', True)
        self.L = self.get_parameter('field_length').value
        self.W = self.get_parameter('field_width').value
        self.G = self.get_parameter('goal_width').value
        self.world = self.get_parameter('world').value
        if self.get_parameter('sim_backend').value == 'mujoco':
            from geometry_msgs.msg import PoseStamped
            self._PoseStamped = PoseStamped
            self.pub_sim_pose = self.create_publisher(PoseStamped, '/sim/set_pose', 10)
        self.r = self.get_parameter('ball_radius').value

        self.tracker = PoseTracker()
        self.yellow = self.blue = 0
        self.cooldown_until = 0.0
        self.last_move_t = time.time()
        self.last_ball = None
        self.out_since = None
        self.kickoff = 'YELLOW'          # team that kicks off next; 'OPEN' once taken
        self.after_goal = False          # True once a goal was awarded -> restarts from the centre spot
        self.last_kickoff = 'YELLOW'
        self.kickoff_spot = None
        self.place_pending = False
        self.kickoff_t0 = 0.0
        self.phase_t0 = time.time()        # when the current phase began
        self.placed_t = None               # kickoff ball placed at (PREPARE)
        self.jam_since = {}                # robot -> time it was first seen inside a goal mouth
        self.restart = ('YELLOW', False)   # (kickoff team, after_goal) of the current restart -> RESET
        self.last_goal = None              # (team, restart before it) -> UNDO_GOAL
        self.goals = []                    # goal log: (mm:ss, team, yellow, blue, kind)
        self.game_t0 = time.time()         # match clock (NEW_GAME)
        self.t0 = time.time()
        self.match_control = self.get_parameter('match_control').value
        self.match_phase = 'WAIT' if self.match_control else 'START'
        self.create_subscription(TFMessage, '/gz/dynamic_pose', self.tracker.update, 10)
        if self.match_control:
            self.create_subscription(String, '/game/control', self._match_control, 10)
        self.pub_control = self.create_publisher(String, '/game/control', 10)
        self.pub_phase = self.create_publisher(String, '/game/phase', 10)
        self.pub_goal = self.create_publisher(String, '/game/goal', 10)      # who scored -> celebration
        self.robot_state = {}                                                # team -> (state, time)
        for team, topic in (('YELLOW', '/yellow/game/state'), ('BLUE', '/blue/game/state'),
                            ('YELLOW', '/game/state')):
            self.create_subscription(String, topic,
                                     lambda m, t=team: self.robot_state.__setitem__(t, (m.data, time.time())), 10)
        self.scorer = None
        self.finish_after_goal = None
        self.pub_goals = self.create_publisher(String, '/game/goals', 10)
        self.pub_score = self.create_publisher(String, '/game/score', 10)
        self.pub_event = self.create_publisher(String, '/game/event', 10)
        self.pub_kickoff = self.create_publisher(String, '/game/kickoff', 10)
        self.pub_initialpose = self.create_publisher(
            PoseWithCovarianceStamped, self.get_parameter('yellow_initialpose_topic').value, 10)
        # blue lives in its own domain: domain_relay forwards /blue/initialpose there
        self.pub_initialpose_blue = self.create_publisher(PoseWithCovarianceStamped, '/blue/initialpose', 10)
        self.create_timer(0.1, self.tick)
        if self.match_control:
            self.create_timer(0.2, self._publish_phase)
        self.get_logger().info(f'referee ready: yellow attacks +X goal, blue attacks -X goal '
                               f'(opening {self.G} m, x=+-{self.L / 2})')
        self._publish_score()

    def _publish_phase(self):
        self.pub_phase.publish(String(data=self.match_phase))
        self.pub_goals.publish(String(data='\n'.join('|'.join(map(str, g)) for g in self.goals)))

    def _publish_score(self):
        self.pub_score.publish(String(data=f'YELLOW {self.yellow} - {self.blue} BLUE'))

    def _event(self, text):
        self.get_logger().info(text)
        self.pub_event.publish(String(data=text))

    # ------------------------------------------------------------------ match flow
    def _set_phase(self, phase):
        """WAIT / PREPARE / START / GOAL / FINISHED; the robots get it on /game/control (FINISHED -> WAIT)."""
        self.match_phase = phase
        self.phase_t0 = time.time()
        self.pub_control.publish(String(data='WAIT' if phase == 'FINISHED' else phase))

    def _match_control(self, msg):
        cmd = msg.data.strip().upper()
        if cmd == 'NEW_GAME':
            self.yellow = self.blue = 0
            self.last_goal = None
            self.goals = []
            self.game_t0 = time.time()
            self._publish_score()
            self.kickoff, self.after_goal = 'YELLOW', False
            self._event('새 경기 시작 - 공은 가운데' if not self.get_parameter('kickoff_rule').value
                        else '새 경기 시작 - 첫 킥오프는 노랑')
            self._prepare()
        elif cmd == 'RESET':
            if self.match_phase in ('WAIT', 'FINISHED'):
                return
            self.kickoff, self.after_goal = self.restart
            self._event(f'원위치 - 로봇은 출발 위치로, {self.kickoff} 킥오프부터 다시 시작')
            self._prepare()
        elif cmd == 'UNDO_GOAL':
            if self.last_goal is None:
                self._event('취소할 골이 없습니다')
                return
            team, restart = self.last_goal
            self.last_goal = None
            if self.goals:
                self.goals.pop()
            if team == 'YELLOW':
                self.yellow = max(0, self.yellow - 1)
            else:
                self.blue = max(0, self.blue - 1)
            self._publish_score()
            self.kickoff, self.after_goal = restart
            self._event(f'{team} 골 취소 -> YELLOW {self.yellow} - {self.blue} BLUE, '
                        f'{self.kickoff} 킥오프부터 다시 시작')
            self._prepare()
        elif cmd == 'FORCE_START':                     # play right now, wherever robots and ball are
            if self.match_phase == 'FINISHED':
                self.yellow = self.blue = 0
                self.last_goal = None
                self.goals = []
                self._publish_score()
            if not self.goals and self.match_phase in ('WAIT', 'FINISHED'):
                self.game_t0 = time.time()
            self.place_pending = False
            self.kickoff = 'OPEN'
            self._event('바로 시작! - 지금 위치에서 경기 진행')
            self._start()
        elif cmd == 'BALL_CENTRE':                     # match window button: only the ball to the centre spot
            self.ball_request_t = time.time()
            if self._centre_ball('공 가운데로 (버튼) - 로봇·점수 그대로, 경기 계속'):
                self.ball_request_t = None
                self.last_ball = None
                self.last_move_t = time.time()
        elif cmd == 'STOP':
            self._set_phase('WAIT')
            self._event('경기 중지')
        # PREPARE / START / GOAL / WAIT are our own outputs (or a test driver): nothing to do here

    def _prepare(self):
        """Both robots to their start poses; the ball goes to the centre, out of their way."""
        if not self.get_parameter('kickoff_rule').value:
            self.kickoff, self.after_goal = 'OPEN', True      # centre spot, nobody waits
        elif self.kickoff not in ('YELLOW', 'BLUE'):
            self.kickoff = self.last_kickoff
        self.restart = (self.kickoff, self.after_goal)
        self.place_pending = True
        self.placed_t = None
        self.kickoff_spot = None
        if self.get_parameter('respot_on_prepare').value:
            self._respot()
        self.last_ball = None
        self._set_phase('PREPARE')

    def _robots_ready(self):
        start = self.L / 2 - self.get_parameter('ready_offset').value
        tol, ytol = self.get_parameter('ready_pos_tol').value, self.get_parameter('ready_yaw_tol').value
        y, b = self.tracker.get('pinky_yellow'), self.tracker.get('pinky_blue')
        if y is None or b is None:
            return False
        if not (math.hypot(y[0] + start, y[1]) < tol and abs(_wrap(y[3])) < ytol
                and math.hypot(b[0] - start, b[1]) < tol and abs(_wrap(b[3] - math.pi)) < ytol):
            return False
        if self.get_parameter('require_robot_ready').value:
            now = time.time()
            for team in ('YELLOW', 'BLUE'):
                st = self.robot_state.get(team)
                if st is None or st[0] != 'READY' or now - st[1] > 2.0:
                    return False
        return True

    def _tick_prepare(self, now):
        timeout = now - self.phase_t0 > self.get_parameter('prepare_timeout').value
        if self.placed_t is None:
            if self._robots_ready() or timeout:
                if timeout:
                    self._event('출발 위치 도착 시간 초과 - 그대로 킥오프')
                self._try_place_kickoff_ball()
                if not self.place_pending:
                    self.placed_t = now
        elif now - self.placed_t > self.get_parameter('start_delay').value:
            self._start()

    def _start(self):
        if self.kickoff_spot is None:
            ball = self.tracker.get('golf_ball')
            self.kickoff_spot = (ball[0], ball[1]) if ball else None
        self.kickoff_t0 = time.time()
        self.last_ball = None
        self.last_move_t = time.time()
        self.cooldown_until = time.time() + 0.5
        self._set_phase('START')
        if self.kickoff != 'OPEN':
            self._event(f'경기 시작 - {self.kickoff} 킥오프')
        elif not self.goals:
            self._event('경기 시작 - 가운데 공, 노랑 선공 (파랑은 3초 뒤 출발)')
        else:
            self._event('경기 시작 - 가운데 공 (직전 득점 팀은 3초 뒤 출발)')

    def _goal(self, team, now):
        """Counted automatically; the conceding team kicks off from the centre spot."""
        self.last_goal = (team, self.restart)
        mine_before, theirs = (self.yellow, self.blue) if team == 'YELLOW' else (self.blue, self.yellow)
        if team == 'YELLOW':
            self.yellow += 1
        else:
            self.blue += 1
        mine = mine_before + 1
        win = self.get_parameter('win_score').value
        if win > 0 and mine >= win:
            kind = '결승골'
        elif self.yellow + self.blue == 1:
            kind = '선제골'
        elif mine == theirs:
            kind = '동점골'
        elif mine < theirs:
            kind = '추격골'
        elif mine_before == theirs:                       # tied before -> now ahead
            kind = '역전골' if self._was_behind(team) else '앞서가는 골'
        else:
            kind = '추가골'
        t = int(now - self.game_t0)
        self.goals.append((f'{t // 60:02d}:{t % 60:02d}', team, self.yellow, self.blue, kind))
        self._publish_score()
        self.kickoff = ('BLUE' if team == 'YELLOW' else 'YELLOW') if self.get_parameter('kickoff_rule').value \
            else 'OPEN'
        self.after_goal = True
        name = '노랑' if team == 'YELLOW' else '파랑'
        line = f'{self.goals[-1][0]}  {name} {kind}!  {self.yellow} : {self.blue}'
        # celebration: the scorer drives to the middle of its own half and spins; GOAL lasts until it is done
        self.scorer = team
        self.robot_state.pop(team, None)
        self.pub_goal.publish(String(data=team))
        self._set_phase('GOAL')
        if win > 0 and max(self.yellow, self.blue) >= win:
            self.finish_after_goal = f'{line}  -  경기 종료, {name} 승리!'
            self._event(f'{line}  -  {name} 세레모니 후 경기 종료')
        else:
            self.finish_after_goal = None
            self._event(f'{line}  -  {name} 세레모니 (제자리 3바퀴), 그 다음 가운데에서 다시 시작')

    def _celebration_over(self, now):
        """The scorer reported CELEBRATE_DONE (after at least goal_pause), or the timeout ran out."""
        elapsed = now - self.phase_t0
        if elapsed < self.get_parameter('goal_pause').value:
            return False
        st = self.robot_state.get(self.scorer)
        if st and st[0] == 'CELEBRATE_DONE' and now - st[1] < 2.0:
            return True
        if elapsed > self.get_parameter('celebrate_timeout').value:
            return True
        # nobody celebrating (e.g. the scripted blue opponent has no celebration): the old short pause
        return st is None and elapsed > self.get_parameter('goal_pause').value + 3.0

    def _was_behind(self, team):
        """Did `team` trail at some point of this match (goal log)?"""
        for _, _, y, b, _ in self.goals:
            if (y < b) if team == 'YELLOW' else (b < y):
                return True
        return False

    def kickoff_x(self, team):
        """Ball centre x for a kickoff: kickoff_gap in front of the bumper of the robot at its start pose."""
        start = self.L / 2 - self.get_parameter('ready_offset').value
        ahead = self.get_parameter('robot_front').value + self.get_parameter('kickoff_gap').value + self.r
        return -start + ahead if team == 'YELLOW' else start - ahead

    def _check_goal_jam(self, now):
        """The crossbar is 7 cm high: a robot whose nose went under it cannot back out by itself.
        Like a person on the real field, lift it out - onto its own start pose (in front of its own goal,
        facing the field): just in front of the goal it is stuck in would box it in behind the other robot."""
        for name in ('pinky_yellow', 'pinky_blue'):
            p = self.tracker.get(name)
            inside = p is not None and abs(p[0]) > self.L / 2 - 0.08 and abs(p[1]) < self.G / 2 + 0.06
            if not inside:
                self.jam_since.pop(name, None)
                continue
            # only a robot that really cannot move: one turning away after a goal right at the line is not stuck
            # (it was lifted out while starting its celebration, r2s3)
            ref = self.jam_since.get(name)
            if ref is None or math.hypot(p[0] - ref[1], p[1] - ref[2]) > 0.03 or abs(_wrap(p[3] - ref[3])) > 0.3:
                self.jam_since[name] = (now, p[0], p[1], p[3])
                continue
            if now - ref[0] < self.get_parameter('jam_timeout').value:
                continue
            self.jam_since.pop(name, None)
            start = self.L / 2 - self.get_parameter('ready_offset').value
            x, y, yaw = (-start, 0.0, 0.0) if name == 'pinky_yellow' else (start, 0.0, math.pi)
            # MuJoCo: drop it from the spawn height (3 cm) - at 1 cm the wheels sink into the floor and the
            # robot stays tilted (lidar on the floor -> AMCL lost, barely drives)
            z = 0.03 if self.get_parameter('sim_backend').value == 'mujoco' else 0.01
            self._set_model_pose(name, x, y, z, yaw)
            # AMCL would keep the old pose: tell it where it is
            pub = self.pub_initialpose if name == 'pinky_yellow' else self.pub_initialpose_blue
            m = PoseWithCovarianceStamped()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x, m.pose.pose.position.y = x, y
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.pose.covariance[0] = m.pose.covariance[7] = 0.0025
            m.pose.covariance[35] = 0.01
            for delay in (0.8, 1.6):                   # after the simulator has moved it (twice: sure)
                threading.Timer(delay, lambda msg=m, p=pub: p.publish(msg)).start()
            self._event(f'{"노랑" if name == "pinky_yellow" else "파랑"} 로봇이 골대에 끼어 출발 위치로 옮김')

    def _set_model_pose(self, name, x, y, z, yaw):
        """Move a model: Gazebo set_pose service, or MuJoCo /sim/set_pose (mujoco_soccer)."""
        if self.get_parameter('sim_backend').value == 'mujoco':
            m = self._PoseStamped()
            m.header.frame_id = name
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.pub_sim_pose.publish(m)
        else:
            threading.Thread(target=gz_set_pose, args=(self.world, name, x, y, z, yaw), daemon=True).start()

    def _respot(self, x=0.0, y=0.0):
        self._set_model_pose('golf_ball', x, y, self.r + 0.003, 0.0)

    def _try_place_kickoff_ball(self):
        """After a goal: centre spot.  Match start: kickoff_gap in front of the kicking robot's bumper, where
        the robot actually stands (its start pose within the READY tolerance)."""
        if self.after_goal:
            self.place_pending = False
            self.kickoff_spot = (0.0, 0.0)
            self._respot(0.0, 0.0)
            self._event('공을 가운데에 놓음' if self.kickoff == 'OPEN' else
                        f'ball on the centre spot - {self.kickoff} touches first')
            return
        kicker = 'pinky_yellow' if self.kickoff == 'YELLOW' else 'pinky_blue'
        me = self.tracker.get(kicker)
        ahead = self.get_parameter('robot_front').value + self.get_parameter('kickoff_gap').value + self.r
        if me is not None:
            x, y = me[0] + ahead * math.cos(me[3]), me[1] + ahead * math.sin(me[3])
        else:
            x, y = self.kickoff_x(self.kickoff), 0.0
        other = self.tracker.get('pinky_blue' if kicker == 'pinky_yellow' else 'pinky_yellow')
        if other is not None and math.hypot(other[0] - x, other[1] - y) < 0.12:
            return                                                       # other robot on the spot
        self.place_pending = False
        self.kickoff_spot = (x, y)
        self._respot(x, y)
        self._event(f'ball placed {self.get_parameter("kickoff_gap").value * 100:.0f} cm in front of '
                    f'{self.kickoff} for the kickoff')

    def _wedged(self, bx, by):
        """Match rule: the ball is caught in a CORNER (both walls within corner_zone) - the robots cannot get
        behind it there.  Anywhere else the ball is left alone (no resets in the open field)."""
        zone = self.get_parameter('corner_zone').value
        wall_x = self.L / 2 + self.get_parameter('outer_strip').value
        wall_y = self.W / 2 + self.get_parameter('outer_strip').value
        return wall_x - abs(bx) < zone and wall_y - abs(by) < zone

    def _centre_ball(self, why):
        """Put the ball back on the centre spot - unless a robot stands on it right now (try again next tick)."""
        for name in ('pinky_yellow', 'pinky_blue'):
            p = self.tracker.get(name)
            if p is not None and math.hypot(p[0], p[1]) < 0.12:
                return False
        self._respot(0.0, 0.0)
        self._event(why)
        return True

    def tick(self):
        now = time.time()
        req = getattr(self, 'ball_request_t', None)
        if req is not None:                            # button pressed while a robot stood on the centre spot
            if self._centre_ball('공 가운데로 (버튼) - 로봇·점수 그대로, 경기 계속'):
                self.ball_request_t, self.last_ball, self.last_move_t = None, None, now
            elif now - req > 3.0:
                self.ball_request_t = None
                self._event('공 가운데로: 센터에 로봇이 있어서 옮기지 못함 - 다시 눌러 주세요')
        if self.match_control:
            self.pub_kickoff.publish(String(data=self.kickoff))
            if self.match_phase in ('PREPARE', 'START', 'GOAL'):   # GOAL: the scorer may be stuck in there
                self._check_goal_jam(now)
            if self.match_phase == 'PREPARE':
                self._tick_prepare(now)
            elif self.match_phase == 'GOAL' and self._celebration_over(now):
                if self.finish_after_goal:
                    self._set_phase('FINISHED')
                    self._event(self.finish_after_goal)
                    self.finish_after_goal = None
                else:
                    self._prepare()
        if self.match_control and self.match_phase != 'START':
            return
        ball = self.tracker.get('golf_ball')
        if ball is None:
            return
        bx, by = ball[0], ball[1]

        if self.kickoff in ('YELLOW', 'BLUE') and self.match_phase == 'START':
            moved = self.kickoff_spot is None or \
                math.hypot(bx - self.kickoff_spot[0], by - self.kickoff_spot[1]) > \
                self.get_parameter('kickoff_touch').value
            if moved or now - self.kickoff_t0 > self.get_parameter('kickoff_timeout').value:
                self._event(f'kickoff taken by {self.kickoff}' if moved else 'kickoff timeout -> ball open')
                self.last_kickoff, self.kickoff = self.kickoff, 'OPEN'

        limit = self.get_parameter('match_seconds').value
        if limit > 0 and now - self.t0 > limit:
            self._event(f'FULL TIME  YELLOW {self.yellow} - {self.blue} BLUE')
            self.t0 = now
            self.yellow = self.blue = 0
            self._publish_score()

        if now < self.cooldown_until:
            return

        in_mouth = abs(by) < self.G / 2
        scored_by = None
        if in_mouth and bx > self.L / 2 + 0.01:
            scored_by = 'YELLOW'
        elif in_mouth and bx < -self.L / 2 - 0.01:
            scored_by = 'BLUE'

        if scored_by and self.match_control:
            self._goal(scored_by, now)
            return

        if scored_by == 'YELLOW':
            self.yellow += 1
            self._event(f'GOAL! YELLOW scores  ->  YELLOW {self.yellow} - {self.blue} BLUE')
        elif scored_by == 'BLUE':
            self.blue += 1
            self._event(f'GOAL! BLUE scores ->  YELLOW {self.yellow} - {self.blue} BLUE')
        else:
            # behind a goal line but beside the goal: bring it back to the centre and play on
            if abs(bx) > self.L / 2 + 0.01 and not in_mouth:
                self.out_since = self.out_since or now
                if now - self.out_since > self.get_parameter('out_timeout').value:
                    if self._centre_ball('골대 옆으로 나간 공 -> 중앙, 경기 계속'):   # match rule
                        self.out_since = None
                        self.last_move_t = now
                return
            self.out_since = None
            # stuck detection: wedged at a wall / under a robot quickly, anything else after stuck_timeout
            if self.last_ball is None or math.hypot(bx - self.last_ball[0], by - self.last_ball[1]) > 0.02:
                self.last_ball = (bx, by)
                self.last_move_t = now
            else:
                still = now - self.last_move_t
                wedged = self._wedged(bx, by)
                zone = self.get_parameter('wall_zone').value
                on_wall = ((self.L / 2 - abs(bx) < zone + 0.03 and abs(by) > self.G / 2) or   # end line incl.
                           self.W / 2 - abs(by) < zone)                                 # next to a post
                if on_wall and not wedged and still > self.get_parameter('wall_stuck_time').value:
                    if self._centre_ball('공이 라인에 붙어 안 움직임 -> 공만 중앙으로, 경기 계속'):
                        self.last_ball = None
                        self.last_move_t = now
                elif (wedged and still > self.get_parameter('wedge_timeout').value) \
                        or 0 < self.get_parameter('stuck_timeout').value < still:
                    if self._centre_ball('공이 모서리에 걸려 못 빠져나옴 -> 중앙으로' if wedged else
                                         '공이 10초 동안 안 움직임 (막힘) -> 공만 중앙으로, 경기 계속'):
                        self.last_ball = None
                        self.last_move_t = now
            return

        self._publish_score()
        self._respot()
        self.cooldown_until = now + 3.0
        self.last_ball = None


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    rclpy.init()
    node = Referee()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
