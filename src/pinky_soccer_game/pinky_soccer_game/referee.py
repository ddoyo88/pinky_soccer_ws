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
        # the person refereeing decides: the ball is never moved by itself; only a ball over a goal line beside
        # the goal is announced, and the match window's "공 가운데로" button puts it back (true: the old automatic
        # resets - line / corner / 10 s standstill / out - for unattended test runs)
        self.declare_parameter('auto_ball_reset', False)
        # attack / defence game (2026-11-25): one robot attacks, the other defends its goal; the roles swap when
        # the ball crosses the centre line into the attacker's own half; after a goal the conceding team attacks
        self.declare_parameter('game_mode', 'attack_defense')        # or 'free' (both always attack)
        # attack / defence rules (user, 2026-11-26): the referee decides the roles - yellow attacks first, the
        # conceding team after a goal, a swap when the ball goes out (it goes back to the centre), and after
        # attack_time s of attack (the ball stays where it is).  The older automatic swaps are off by default.
        self.declare_parameter('attack_time', 60.0)        # 1 minute per attack (user, back on 2026-11-26)        # s per attack, 0 = no timer - counted from the
                                                           # attacker's first touch of the ball (user, option B)
        self.declare_parameter('swap_on_centre_line', False)
        # trial rule (user, 2026-11-26): the half the ball is in decides - ball in yellow's half -> blue attacks,
        # in blue's half -> yellow attacks (5 cm past the centre line, so it does not flicker on the line)
        self.declare_parameter('swap_by_half', False)
        # every swap (user, 2026-11-27): both robots first go to their own places fast - the attacker to its start
        # pose (or just behind the ball when it lies there), the defender in front of its goal - then play on
        self.declare_parameter('regroup_all', True)   # trial rule, off again
        self.declare_parameter('drill', False)             # attack drill: yellow alone, random balls
        # timed match (user, 2026-11-26): match_time s of play, more goals wins; a draw -> golden goal (the next
        # goal wins).  win_score then does not end the match early.  0 = first to win_score as before
        self.declare_parameter('match_time', 600.0)
        self.declare_parameter('drill_time', 60.0)         # s per drill round
        self.declare_parameter('drill_rounds', 10)
        self.declare_parameter('swap_on_ball_win', False)
        self.declare_parameter('wall_zone', 0.08)          # m: ball this close to the boundary LINE (or beyond
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
        # /game/attack: 'YELLOW' / 'BLUE' = that team attacks, the other defends; '<TEAM>_KICKOFF' = that team
        # goes to the ball, the other waits until it is touched; 'FREE' = both attack (old game)
        self.pub_attack = self.create_publisher(String, '/game/attack', 10)
        self.pub_attack_clock = self.create_publisher(String, '/game/attack_clock', 10)
        self.pub_game_clock = self.create_publisher(String, '/game/clock', 10)   # match time mm:ss
        # automatic start countdown for the match window: 'MOVE <s>' robots on the way to their start poses (start
        # at the latest after <s>), 'START <s>' ball placed, kick-off in <s>; '' otherwise
        self.pub_start_in = self.create_publisher(String, '/game/start_in', 10)
        # the robots' own camera sightings of the ball (relayed): 'nobody can find the ball'
        from geometry_msgs.msg import PointStamped as _PS
        self.robot_ball_t = time.time()
        for topic in ('/yellow/ball/position_map', '/blue/ball/position_map'):
            self.create_subscription(_PS, topic, lambda m: setattr(self, 'robot_ball_t', time.time()), 10)
        self.attack_t0 = time.time()
        self.attack, self.attack_touched, self.next_attack = 'YELLOW', False, 'YELLOW'
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
            self.golden = False
            self._publish_score()
            self.kickoff, self.after_goal = 'YELLOW', False
            self.next_attack = 'YELLOW'                   # first attack: yellow
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
                self.golden = False
            self.place_pending = False
            self.kickoff = 'OPEN'
            self._event('바로 시작! - 지금 위치에서 경기 진행')
            self._start()
        elif cmd == 'SWAP':                            # match window button: the referee swaps the roles
            if self._attack_mode() and self.match_phase == 'START':
                self._attack_turnover()
                self._event('심판 판단: 공수 교대 (공은 그대로)')
        elif cmd == 'BALL_CENTRE':                     # match window button: only the ball to the centre spot
            self.ball_request_t = time.time()
            if self._centre_ball('공 가운데로 (버튼) - 로봇·점수 그대로, 경기 계속'):
                self.ball_request_t = None
                self.last_ball = None
                self.last_move_t = time.time()
                if not getattr(self, 'out_pending', False):
                    self._attack_turnover()          # (an out has swapped the roles already)
                self.out_pending = False
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
        timeout = now - self.phase_t0 > (3.0 if self.get_parameter('drill').value       # drill: blue never gets
                                         else self.get_parameter('prepare_timeout').value)  # ready - start at once
        limit = 3.0 if self.get_parameter('drill').value else self.get_parameter('prepare_timeout').value
        if self.placed_t is None:
            self.pub_start_in.publish(String(data=f'MOVE {max(0.0, limit - (now - self.phase_t0)):.0f}'))
        else:
            self.pub_start_in.publish(String(data='START %.1f' % max(
                0.0, self.get_parameter('start_delay').value - (now - self.placed_t))))
        if self.placed_t is None:
            if self._robots_ready() or timeout:
                if timeout:
                    self._event('출발 위치 도착 시간 초과 - 그대로 킥오프')
                self._try_place_kickoff_ball()
                if not self.place_pending:
                    self.placed_t = now
        elif now - self.placed_t > self.get_parameter('start_delay').value:
            self._start()

    def _attack_mode(self):
        return self.get_parameter('game_mode').value == 'attack_defense'

    def _attack_turnover(self, immediate=True):
        """The ball went out / the referee put it back (button): the defending team won it - the roles swap at
        once (no kick-off wait: the new defender heads for its goal straight away)."""
        if self._attack_mode() and self.match_phase == 'START':
            self.attack = 'BLUE' if self.attack == 'YELLOW' else 'YELLOW'
            self.attack_touched = immediate
            self.half_prev = 0
            self.swap_t = time.time()
            self.regroup_t0 = None
            self.attack_t0 = time.time()                 # a fresh attack time for the new attacker ...
            self.clock_running = False                   # ... starting with its first touch of the ball
            if self.get_parameter('regroup_all').value:
                self.regroup_t0 = time.time()            # first to their places, then play
            other = '노랑' if self.attack == 'BLUE' else '파랑'
            self._event(f'공수 교대: {"노랑" if self.attack == "YELLOW" else "파랑"} 공격 / {other} 수비')

    def _attack_tick(self, bx, by):
        """Kick-off touch, then the roles swap whenever the ball crosses the centre line into the attacker's own
        half (yellow attacks +x, blue -x; 5 cm hysteresis)."""
        name = {'YELLOW': '노랑', 'BLUE': '파랑'}
        if not self.attack_touched:
            p = self.tracker.get('pinky_yellow' if self.attack == 'YELLOW' else 'pinky_blue')
            touched = math.hypot(bx, by) > 0.06 or (p is not None and math.hypot(p[0] - bx, p[1] - by) < 0.13)
            if touched:
                self.attack_touched = True
                self.attack_t0 = time.time()
                self.clock_running = True
                other = 'BLUE' if self.attack == 'YELLOW' else 'YELLOW'
                self._event(f'{name[self.attack]} 공격 / {name[other]} 수비')
            return
        if abs(bx) > self.L / 2 + 0.01 or abs(by) > self.W / 2 + 0.01:
            return                       # ball out: the out rule has swapped the roles already (r: ad2 double swap)
        at = self.get_parameter('attack_time').value
        if getattr(self, 'regroup_t0', None) is not None:
            return                       # both robots on their way to their own half
        if self.get_parameter('swap_by_half').value:
            want = 'BLUE' if bx < -0.05 else ('YELLOW' if bx > 0.05 else self.attack)
            if want != self.attack:
                self._attack_turnover()
                self._event(f'공이 {"노랑" if want == "BLUE" else "파랑"} 진영으로 - '
                            f'{"파랑" if want == "BLUE" else "노랑"} 공격')
                return
        if at > 0 and not getattr(self, 'clock_running', True):
            # the attack clock starts at the attacker's first touch (within 0.14 m of the ball); before that the
            # time it needs to get to the ball does not count
            ap = self.tracker.get('pinky_yellow' if self.attack == 'YELLOW' else 'pinky_blue')
            if ap is not None and math.hypot(ap[0] - bx, ap[1] - by) < 0.14:
                self.clock_running, self.attack_t0 = True, time.time()
            else:
                return
        if at > 0 and time.time() - self.attack_t0 > at and not getattr(self, 'out_pending', False):
            # attack time up with the ball in the field: the ball stays, the roles swap - the attacker runs back
            # to its goal at full speed, the defender takes the ball on
            # the ball stays where it is - unless nobody can find it, or it lies still next to a goal or on a
            # line (user, 2026-11-26): then it goes back to the centre spot
            now_t = time.time()
            lost = now_t - self.robot_ball_t > 3.0
            still = self.last_move_t is not None and now_t - self.last_move_t > 2.0
            near_goal = min(math.hypot(bx - gx, by) for gx in (-self.L / 2, self.L / 2)) < 0.30
            on_line = self.W / 2 - abs(by) < 0.06 or (self.L / 2 - abs(bx) < 0.06 and abs(by) > self.G / 2 - 0.03)
            self._attack_turnover()
            if lost or (still and (near_goal or on_line)):
                why = '공을 찾을 수 없음' if lost else ('골대 근처에 멈춤' if near_goal else '라인에 멈춤')
                if self._centre_ball(f'공격 시간 {at:.0f}초 끝 ({why}) - 공을 가운데로, 공수 교대'):
                    self.last_ball, self.last_move_t = None, now_t
                    self.attack_t0, self.clock_running = now_t, False
                    return
            # the ball stays: both robots first go back to their own half (the new attacker to its start pose,
            # or just behind the ball when that lies there), then play on (user, 2026-11-26)
            self.regroup_t0 = now_t
            self._event(f'공격 시간 {at:.0f}초 끝 - 공은 그대로, 공수 교대: 각자 자기 진영으로')
            return
        if not self.get_parameter('swap_on_ball_win').value and not self.get_parameter('swap_on_centre_line').value:
            return
        # the defender wins the ball: the ball at its nose, moving away from its goal, for 0.7 s -> it attacks
        # now and the attacker has to run back at once (user: 'yellow should hurry back to its goal')
        dteam = 'BLUE' if self.attack == 'YELLOW' else 'YELLOW'
        dp = self.tracker.get('pinky_blue' if dteam == 'BLUE' else 'pinky_yellow')
        away = 1.0 if dteam == 'YELLOW' else -1.0       # yellow's goal is at -x: away = +x
        prev = getattr(self, 'poss_ball', None)
        self.poss_ball = (bx, by)
        moving_away = prev is not None and (bx - prev[0]) * away > 0.002
        settled = time.time() - getattr(self, 'swap_t', 0.0) > 3.0      # no ping-pong: 3 s after any swap
        if self.get_parameter('swap_on_ball_win').value and dp is not None and \
                math.hypot(dp[0] - bx, dp[1] - by) < 0.14 and moving_away and settled:
            self.poss_t = getattr(self, 'poss_t', None) or time.time()
            if time.time() - self.poss_t > 1.0:
                self.poss_t = None
                self.attack = dteam
                self.half_prev = 0
                self.swap_t = time.time()
                other = 'BLUE' if dteam == 'YELLOW' else 'YELLOW'
                self._event(f'공수 교대 (공 뺏음): {name[dteam]} 공격 / {name[other]} 수비')
                return
        elif dp is None or math.hypot(dp[0] - bx, dp[1] - by) > 0.20:
            self.poss_t = None
        # only a CROSSING of the centre line (from the attacking half into the attacker's own half) swaps - a
        # ball already lying in the new attacker's half after a ball win must not swap straight back (ad6)
        side = 1 if bx > 0.05 else (-1 if bx < -0.05 else 0)
        own = -1 if self.attack == 'YELLOW' else 1          # the attacker's own half
        prev = getattr(self, 'half_prev', 0)
        self.half_prev = side if side != 0 else prev
        back = side == own and prev == -own
        if back and self.get_parameter('swap_on_centre_line').value:
            self.swap_t = time.time()
            self.attack = 'BLUE' if self.attack == 'YELLOW' else 'YELLOW'
            other = 'BLUE' if self.attack == 'YELLOW' else 'YELLOW'
            self._event(f'공수 교대: {name[self.attack]} 공격 / {name[other]} 수비')

    def _start(self):
        if self.kickoff_spot is None:
            ball = self.tracker.get('golf_ball')
            self.kickoff_spot = (ball[0], ball[1]) if ball else None
        self.kickoff_t0 = time.time()
        self.attack, self.attack_touched = self.next_attack, False
        self.clock_running = False
        self.regroup_t0 = None
        self.half_prev = 0
        self.last_ball = None
        self.last_move_t = time.time()
        self.cooldown_until = time.time() + 0.5
        self._set_phase('START')
        if self.kickoff != 'OPEN':
            self._event(f'경기 시작 - {self.kickoff} 킥오프')
        elif not self.goals:
            self._event('경기 시작 - 가운데 공, 노랑 선공 (파랑은 2초 뒤 출발)')
        else:
            self._event('경기 시작 - 가운데 공 (직전 득점 팀은 2초 뒤 출발)')

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
        self.next_attack = 'BLUE' if team == 'YELLOW' else 'YELLOW'     # the conceding team attacks next
        name = '노랑' if team == 'YELLOW' else '파랑'
        line = f'{self.goals[-1][0]}  {name} {kind}!  {self.yellow} : {self.blue}'
        # celebration: the scorer drives to the middle of its own half and spins; GOAL lasts until it is done
        self.scorer = team
        self.robot_state.pop(team, None)
        self.pub_goal.publish(String(data=team))
        self._set_phase('GOAL')
        timed = self.get_parameter('match_time').value > 0
        if (timed and getattr(self, 'golden', False)) or (not timed and win > 0 and max(self.yellow, self.blue) >= win):
            self.finish_after_goal = f'{line}  -  {"골든골! " if timed else ""}경기 종료, {name} 승리!'
            self._event(f'{line}  -  {name} 세레모니 후 경기 종료')
        else:
            self.finish_after_goal = None
            self._event(f'{line}  -  {name} 세레모니 (제자리 1바퀴), 그 다음 가운데에서 다시 시작')

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
            inside = p is not None and abs(p[0]) > self.L / 2 - 0.04 and abs(p[1]) < self.G / 2 + 0.06   # (0.08 lifted keepers standing in front of their goal, final5i)
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
            self._lift_robot(name, f'{"노랑" if name == "pinky_yellow" else "파랑"} 로봇이 골대에 끼어 출발 위치로 옮김')

    def _check_robot_lock(self, now):
        """Two robots hooked into each other (g23, 2026-11-26: 8 min without moving a millimetre while both
        tried to drive out): like a person on the field, separate them - the defender (blue in a free game)
        is put back on its start pose in front of its own goal, where it was heading anyway."""
        y, b = self.tracker.get('pinky_yellow'), self.tracker.get('pinky_blue')
        if y is None or b is None or math.hypot(y[0] - b[0], y[1] - b[1]) > 0.18:
            self.lock_ref = None
            return
        ref = getattr(self, 'lock_ref', None)
        moved = ref is not None and any(
            math.hypot(p[0] - r[0], p[1] - r[1]) > 0.03 or abs(_wrap(p[3] - r[3])) > 0.3
            for p, r in ((y, ref[1]), (b, ref[2])))
        if ref is None or moved:
            self.lock_ref = (now, y, b)
            return
        if now - ref[0] < 8.0:
            return
        self.lock_ref = None
        defender = 'pinky_blue'
        if self._attack_mode() and getattr(self, 'attack', 'YELLOW') == 'BLUE':
            defender = 'pinky_yellow'
        self._lift_robot(defender, f'두 로봇이 엉켜 못 움직임 - {"노랑" if defender == "pinky_yellow" else "파랑"}(수비)을 골대 앞으로 옮김')

    def _lift_robot(self, name, text):
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
        def send(msg=m, p=pub):
            # stamp NOW: AMCL moves an initial pose by the odometry since its stamp - with the rescue-time
            # stamp that included the teleport, so AMCL started 2 rad off and flipped to the mirrored
            # field (r8s2: 1.4 m error for the rest of the set)
            msg.header.stamp = self.get_clock().now().to_msg()
            p.publish(msg)
        for delay in (0.8, 1.6):                   # after the simulator has moved it (twice: sure)
            threading.Timer(delay, send).start()
        self._event(text)

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

    def _drill_tick(self, now, bx, by):
        """Attack drill (user, 2026-11-26): yellow alone (blue parked outside the field, no striker), the ball
        dropped at a random spot, yellow has drill_time s to score in blue's goal.  Goal = success, time up or
        ball out = miss; then the next random ball.  Results go to the event log ('연습 N: ...')."""
        import random
        self.attack, self.attack_touched, self.regroup_t0 = 'YELLOW', True, None
        if not getattr(self, 'drill_parked', False):
            self.drill_parked = True
            self._set_model_pose('pinky_blue', 0.0, -(self.W / 2 + self.get_parameter('outer_strip').value * 0.6),
                                 0.03, 0.0)
            self.drill_n, self.drill_ok, self.drill_times = 0, 0, []
            self.drill_t0 = None
        # rounds of drill_time s (user, 2026-11-26): a random ball at the start of each round; a goal or an out ->
        # the next random ball at once (1 s pause), the clock keeps running; count the goals per round
        rounds = int(self.get_parameter('drill_rounds').value)
        limit = self.get_parameter('drill_time').value
        if getattr(self, 'drill_done', False):
            return
        if self.drill_t0 is None:                           # a new round
            self.drill_t0, self.drill_goals = now, 0
            self.drill_n += 1
            self.drill_next = now                            # ball now
            self._event(f'연습 {self.drill_n}/{rounds} 시작 - {limit:.0f}초 동안 몇 골?')
        if self.drill_next is not None:
            if now < self.drill_next:
                return
            yp = self.tracker.get('pinky_yellow')
            for _ in range(50):                             # random spot inside the field, away from the robot
                x = random.uniform(-self.L / 2 + 0.15, self.L / 2 - 0.15)
                y = random.uniform(-self.W / 2 + 0.12, self.W / 2 - 0.12)
                if yp is None or math.hypot(x - yp[0], y - yp[1]) > 0.30:
                    break
            self._respot(x, y)
            self.drill_next, self.drill_ball_t = None, now
            return
        if now - self.drill_t0 > limit:                     # round over
            self.drill_times.append(self.drill_goals)
            total = sum(self.drill_times)
            self._event(f'연습 {self.drill_n}/{rounds} 끝: {self.drill_goals}골  -  지금까지 {total}골 '
                        f'({self.drill_n}판, 판당 {total / self.drill_n:.1f}골)')
            if self.drill_n >= rounds:
                self.drill_done = True
                self._event(f'연습 종료: {rounds}판 {total}골, 판별 {self.drill_times}')
                return
            self.drill_t0 = None
            return
        if now - self.drill_ball_t < 1.0:
            return                                          # (the ball position lags the re-spot)
        goal = bx > self.L / 2 + 0.01 and abs(by) < self.G / 2
        out = not goal and (abs(bx) > self.L / 2 + 0.01 or abs(by) > self.W / 2 + 0.01)
        # a ball stuck on a line for 5 s (no robot can get behind it without crossing the line) -> dropped again,
        # like the referee button in a match
        on_line = self.W / 2 - abs(by) < 0.06 or (self.L / 2 - abs(bx) < 0.06 and abs(by) > self.G / 2)
        ref = getattr(self, 'drill_line_ref', None)
        if not on_line or ref is None or math.hypot(bx - ref[0], by - ref[1]) > 0.02:
            self.drill_line_ref = (bx, by, now)
        elif now - ref[2] > 5.0:
            self.drill_line_ref = None
            self._event('공이 라인에 붙어 꺼낼 수 없음 - 다시 놓음')
            self.drill_next = now + 1.0
            return
        if goal or out:
            if goal:
                self.drill_goals += 1
                self.drill_ok += 1
                self._event(f'골! (연습 {self.drill_n}: {self.drill_goals}골째, {now - self.drill_t0:.0f}초)')
            else:
                self._event('공이 밖으로 나감 - 다시 놓음')
            self.drill_next = now + 1.0                     # the next random ball

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
        self.closer_pending = True          # next tick: the robot closer to the ball takes it (after any swap)
        return True

    def _regroup_tick(self, now):
        """Regroup after a time-up swap: over when the attacker stands on its spot and the defender is in its own
        half (or after 10 s); the attack clock then waits for the attacker's first touch as usual."""
        from pinky_soccer_game.ball_play import regroup_spot
        if self.match_phase != 'START':
            self.regroup_t0 = None
            return
        att = 'pinky_yellow' if self.attack == 'YELLOW' else 'pinky_blue'
        dfn = 'pinky_blue' if self.attack == 'YELLOW' else 'pinky_yellow'
        a, d, ball = self.tracker.get(att), self.tracker.get(dfn), self.tracker.get('golf_ball')
        own_x = -self.L / 2 if self.attack == 'YELLOW' else self.L / 2
        ok = False
        if a is not None and d is not None and ball is not None:
            sx, sy = regroup_spot(own_x, (ball[0], ball[1]), self.L / 2, self.W / 2,
                                  self.get_parameter('ready_offset').value)
            d_own = -self.L / 2 if dfn == 'pinky_yellow' else self.L / 2
            ok = math.hypot(a[0] - sx, a[1] - sy) < 0.08 and abs(d[0] - d_own) < 0.60
        if ok or now - self.regroup_t0 > 12.0:
            self.regroup_t0 = None
            self.attack_t0, self.clock_running = now, False
            name = {'YELLOW': '노랑', 'BLUE': '파랑'}
            self._event(f'자리 잡기 끝 - {name[self.attack]} 공격 시작')

    def _attack_to_closer(self):
        """Ball (re)placed on the centre spot (user, 2026-11-26): the robot closer to it goes to the ball (attacks),
        the one further away backs off (defends, back to its goal).  Within 5 cm of a tie the roles stay as the
        other rules set them (after a goal both stand on their start poses: the conceding team attacks)."""
        if not self._attack_mode() or self.match_phase != 'START' or self.get_parameter('regroup_all').value:
            return                                       # (regroup_all: everybody goes to its place instead)
        y, b = self.tracker.get('pinky_yellow'), self.tracker.get('pinky_blue')
        if y is None or b is None:
            return
        dy, db = math.hypot(y[0], y[1]), math.hypot(b[0], b[1])
        if abs(dy - db) < 0.05:
            return
        closer = 'YELLOW' if dy < db else 'BLUE'
        if closer != self.attack:
            self.attack = closer
            self.attack_touched = True
            self.half_prev = 0
            self.swap_t = self.attack_t0 = time.time()
            self.clock_running = False
            name = {'YELLOW': '노랑', 'BLUE': '파랑'}
            other = 'BLUE' if closer == 'YELLOW' else 'YELLOW'
            self._event(f'가운데 공: 가까운 {name[closer]} 공격 / 먼 {name[other]} 수비 (뒤로)')

    def _check_match_time(self, now):
        """Timed match: at match_time the team ahead wins; a draw goes on as golden goal."""
        mt = self.get_parameter('match_time').value
        if mt <= 0 or self.match_phase != 'START' or getattr(self, 'golden', False) or now - self.game_t0 < mt:
            return
        if self.yellow != self.blue:
            win = '노랑' if self.yellow > self.blue else '파랑'
            self._set_phase('FINISHED')
            self._event(f'{int(mt // 60)}분 종료  {self.yellow} : {self.blue}  -  경기 종료, {win} 승리!')
        else:
            self.golden = True
            self._event(f'{int(mt // 60)}분 종료  {self.yellow} : {self.blue} 동점  -  골든골! 다음 골을 넣는 팀이 승리')

    def _publish_game_clock(self, now):
        """Match time (mm:ss since NEW_GAME) for the match window and the video; stands still once finished."""
        ph = self.match_phase
        if ph in ('WAIT',):
            text = '00:00'
        elif ph == 'FINISHED':
            text = getattr(self, 'clock_text', '00:00')
        else:
            t = int(max(0.0, now - self.game_t0))
            text = f'{t // 60:02d}:{t % 60:02d}'
        self.clock_text = text
        self.pub_game_clock.publish(String(data=text))

    def tick(self):
        now = time.time()
        self._publish_game_clock(now)
        self._check_match_time(now)
        if getattr(self, 'closer_pending', False):
            self.closer_pending = False
            self._attack_to_closer()
        req = getattr(self, 'ball_request_t', None)
        if req is not None:                            # button pressed while a robot stood on the centre spot
            if self._centre_ball('공 가운데로 (버튼) - 로봇·점수 그대로, 경기 계속'):
                self.ball_request_t, self.last_ball, self.last_move_t = None, None, now
                if not getattr(self, 'out_pending', False):
                    self._attack_turnover()          # (an out has swapped the roles already)
                self.out_pending = False
            elif now - req > 3.0:
                self.ball_request_t = None
                self._event('공 가운데로: 센터에 로봇이 있어서 옮기지 못함 - 다시 눌러 주세요')
        if self.match_control:
            self.pub_kickoff.publish(String(data=self.kickoff))
            if getattr(self, 'regroup_t0', None) is not None:
                self._regroup_tick(now)
            regroup = getattr(self, 'regroup_t0', None) is not None
            self.pub_attack.publish(String(data='FREE' if not self._attack_mode() else
                                           self.attack + ('_REGROUP' if regroup else
                                                          '' if self.attack_touched else '_KICKOFF')))
            if self._attack_mode() and self.match_phase == 'START':
                at = self.get_parameter('attack_time').value
                running = at > 0 and self.attack_touched and getattr(self, 'clock_running', True)
                left = at - (now - self.attack_t0) if running else at
                if at <= 0:
                    self.pub_attack_clock.publish(String(data=f'{self.attack} -1'))      # no attack timer
                else:
                    self.pub_attack_clock.publish(String(data=f'{self.attack} {left:.1f}' + ('' if running else ' WAIT')))
            if self.match_phase in ('PREPARE', 'START', 'GOAL'):   # GOAL: the scorer may be stuck in there
                self._check_goal_jam(now)
            if self.match_phase == 'START':
                self._check_robot_lock(now)
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
        if self.get_parameter('drill').value and self.match_phase == 'START':
            self._drill_tick(now, bx, by)
            return
        if self._attack_mode() and self.match_phase == 'START':
            self._attack_tick(bx, by)

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
            if not self.get_parameter('auto_ball_reset').value or self._attack_mode():
                auto = self.get_parameter('auto_ball_reset').value
                out = (abs(bx) > self.L / 2 + 0.01 and not in_mouth) or abs(by) > self.W / 2 + 0.01
                pending = getattr(self, 'out_pending', False)
                if not out and abs(bx) < self.L / 2 - 0.05 and abs(by) < self.W / 2 - 0.05:
                    self.out_armed = True        # seen back in the field: the next out counts
                if out and not pending and getattr(self, 'out_armed', True):
                    where = '골라인' if abs(bx) > self.L / 2 + 0.01 else '사이드라인'
                    self.out_t, self.out_pending, pending = now, True, True
                    self.out_armed = False       # (the ball position lags the re-spot: no second out from it)
                    # out: the roles swap at once (the attacker turns back to defend right away), the ball goes
                    # back to the centre by itself after 1 s (user, 2026-11-25) - unclear cases stay the button's.
                    # One out = one swap: a ball bouncing on the line counted twice and swapped back (ad3)
                    self._attack_turnover(immediate=True)
                    self._event(f'공이 {where} 밖으로 나감 - 1초 뒤 가운데로')
                if pending and now - self.out_t > 1.0:
                    if self._centre_ball('나간 공 -> 가운데, 경기 계속'):
                        self.last_ball, self.last_move_t = None, now
                        self.attack_t0 = time.time()       # (the clock waits for the new attacker's touch)
                        self.clock_running = False
                        self.out_pending = False
                        out = False
                self.out_called = out
                # ball lying on a line where no robot can reach it without crossing the line: only an
                # announcement - the person refereeing decides (button), the ball is not moved
                if self.last_ball is None or math.hypot(bx - self.last_ball[0], by - self.last_ball[1]) > 0.02:
                    self.last_ball, self.last_move_t, self.line_called = (bx, by), now, False
                elif not out and not getattr(self, 'line_called', False) and now - self.last_move_t > 5.0 and \
                        (self.L / 2 - abs(bx) < 0.06 and abs(by) > self.G / 2 - 0.03 or self.W / 2 - abs(by) < 0.06):
                    self.line_called = True
                    self._event('공이 라인에 붙어 로봇이 꺼낼 수 없음 - 심판 판단: "공 가운데로" 버튼')
                # unattended test runs (auto_ball_reset): what the person would do - the stuck ball (5 s on a line,
                # 15 s anywhere) back to the centre, the roles swap like after the button
                # (the 15 s 'still anywhere' reset is gone: the attack timer swaps the roles instead)
                if auto and not out and self.last_move_t is not None and (
                        (getattr(self, 'line_called', False) and now - self.last_move_t > 5.0) or
                        now - self.last_move_t > 45.0):
                    if self._centre_ball('(자동 테스트) 멈춘 공 -> 가운데, 공수 교대'):
                        self.last_ball, self.last_move_t, self.line_called = None, now, False
                        self._attack_turnover()
                return
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
                on_wall = ((self.L / 2 - abs(bx) < zone + 0.05 and abs(by) > self.G / 2 - 0.05) or   # end line incl.
                           self.W / 2 - abs(by) < zone)          # side line; end line up to 5 cm inside a post (r7s2)
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
