#!/usr/bin/env python3
"""Referee for the REAL field (one yellow robot): works from the camera's ball estimate (/ball/position_map),
not from simulator ground truth.  Same topics as referee.py, so match_ui and the striker work unchanged.

Flow (buttons of the match UI on /game/control: NEW_GAME / RESET / UNDO_GOAL / STOP):
  NEW_GAME -> PREPARE: the robot drives to its start pose in front of its own goal (-X)
  robot READY + ball lying still ON THE CENTRE SPOT (`centre_tol`) for `ball_still_time` -> START
  (a person puts the ball down; anywhere else: no auto-start - use FORCE_START / 바로 시작)
  goal: ball seen behind a goal line inside the mouth, or last seen right at the +X mouth and then lost
        (it rolled in - the robot's nose may hide it) -> score, GOAL, after `goal_pause` PREPARE again:
        the robot goes back in front of its own goal and waits for the ball to be put back
  corner: ball in a corner (both walls within `corner_zone`) for `corner_time` -> CORNER: sound, robot back
        to its start pose, a person puts the ball back (same restart as after a goal)
  pose guard: AMCL puts the robot outside the walls (wheels slipped against a wall, odometry ran on)
        -> /initialpose at the nearest point inside, so AMCL re-converges instead of planning from outside
  /game/fx (String) GOAL / CORNER / FINISHED -> pinky_effects.py on the robot: LED blinking + buzzer
  first to `win_score` -> FINISHED.  RESET: back to the start pose.  UNDO_GOAL: take the last goal back.
  FORCE_START: play right now, wherever the robot and the ball are (a finished game starts again at 0:0).
"""
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import String

from pinky_soccer_sim.field import default_dims

FIELD_L, FIELD_W, FIELD_G, FIELD_STRIP = default_dims()


class RealReferee(Node):
    def __init__(self):
        super().__init__('referee')
        P = self.declare_parameter
        P('field_length', FIELD_L)
        P('field_width', FIELD_W)
        P('goal_width', FIELD_G)
        P('win_score', 5)
        P('duel', False)                # two real robots: blue's state / ball come relayed from its domain
                                        # (/blue/game/state, /blue/ball/position_map - domain_relay)
        P('celebrate_timeout', 60.0)    # s max for the celebration (drive + 3 turns)
        P('goal_pause', 2.0)            # s standing still after a goal
        P('ball_still_time', 2.0)       # s the placed ball must lie still before the kickoff
        P('ball_still_tol', 0.04)       # m
        P('place_ball_sim', False)      # simulation: put the ball on the centre spot (like the person on the
                                        # real field) once the robot is READY
        P('world', 'soccer')
        P('sim_backend', 'gz')          # gz | mujoco
        P('ball_radius', 0.02135)
        P('centre_tol', 0.12)           # m: auto-start only with the ball on the centre spot
        P('goal_hits', 4)               # estimates behind a goal line needed ...
        P('goal_window', 0.6)           # ... within this many seconds -> goal (one stray estimate is not)
        P('lost_goal_time', 2.5)        # s ball unseen after being at the +X mouth -> it went in
        P('mouth_margin', 0.08)         # m in front of the goal line that counts as 'at the mouth'
        P('outer_strip', FIELD_STRIP)   # field line -> wall
        P('corner_zone', 0.22)          # m from both walls = in the corner (the robot cannot get behind it)
        P('corner_time', 1.0)           # s
        P('corner_pause', 1.5)          # s robot stands (sound) before going back
        P('wall_zone', 0.07)            # m: ball this close to the boundary line (or beyond; robots must not
                                        # cross the line - goal mouths excepted) ...
        P('wall_stuck_time', 10.0)      # ... lying still this long -> ask for the ball on the centre spot
        P('stuck_time', 20.0)           # ball not moving at all this long anywhere (deadlock) -> same
                                        # (robots play on: nobody can get behind a ball on the wall)
        self.L = self.get_parameter('field_length').value
        self.W = self.get_parameter('field_width').value
        self.G = self.get_parameter('goal_width').value
        self.phase = 'WAIT'
        self.phase_t0 = time.time()
        self.yellow = self.blue = 0
        self.goals = []
        self.game_t0 = time.time()
        self.ball = None                # (x, y, t)
        self.still_ref = None           # (x, y, t) ball position where it started lying still
        self.still_src = {}             # per camera: (x, y, t) - two cameras see the ball a few cm apart,
                                        # so mixing them would never count as 'lying still'
        self.wait_note_t = 0.0
        self.robot_state, self.robot_state_t = '', 0.0
        self.asked_ball = False
        self.ball_placed = False
        self.last_out = 0.0
        self.ball_at_mouth_t = None
        self.corner_since = None
        self.armed = False              # goals count only after the ball was seen on the field since START
        self.strip = self.get_parameter('outer_strip').value

        self.pub_control = self.create_publisher(String, '/game/control', 10)
        self.pub_phase = self.create_publisher(String, '/game/phase', 10)
        self.pub_score = self.create_publisher(String, '/game/score', 10)
        self.pub_goals = self.create_publisher(String, '/game/goals', 10)
        self.pub_event = self.create_publisher(String, '/game/event', 10)
        self.pub_kickoff = self.create_publisher(String, '/game/kickoff', 10)
        self.pub_fx = self.create_publisher(String, '/game/fx', 10)
        self.pub_goal = self.create_publisher(String, '/game/goal', 10)
        self.celebration_seen = False
        self.finish_after_goal = None
        if self.get_parameter('sim_backend').value == 'mujoco':
            from geometry_msgs.msg import PoseStamped
            self.pub_sim_pose = self.create_publisher(PoseStamped, '/sim/set_pose', 10)
        self.create_subscription(String, '/game/control', self._control, 10)
        self.create_subscription(String, '/game/state', self._robot_state, 10)
        self.create_subscription(PointStamped, '/ball/position_map', lambda m: self._ball(m, 'YELLOW'), 10)
        self.duel = self.get_parameter('duel').value
        self.ball_src = {}                # source robot -> (x, y, t) latest accepted estimate
        self.goal_hits = []               # (scoring team, source robot, t) estimates behind a goal line
        self.states = {'YELLOW': ('', 0.0), 'BLUE': ('', 0.0)}
        self.scorer = 'YELLOW'
        if self.duel:                   # either robot's camera may see the ball: use the latest estimate
            self.create_subscription(String, '/blue/game/state',
                                     lambda m: self._team_state('BLUE', m.data), 10)
            self.create_subscription(PointStamped, '/blue/ball/position_map', lambda m: self._ball(m, 'BLUE'), 10)
        self.pub_initialpose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl, 10)
        self.outside_since = None
        self.last_reset = 0.0
        self.create_timer(0.1, self.tick)
        self.create_timer(0.2, self._publish_status)
        self.get_logger().info('real referee ready (ball from the camera, yellow attacks +X)')

    # ------------------------------------------------------------------ io
    def _publish_status(self):
        self.pub_phase.publish(String(data=self.phase))
        self.pub_score.publish(String(data=f'YELLOW {self.yellow} - {self.blue} BLUE'))
        self.pub_goals.publish(String(data='\n'.join('|'.join(map(str, g)) for g in self.goals)))
        self.pub_kickoff.publish(String(data='OPEN'))      # one robot: no kickoff waiting

    def _event(self, text):
        self.get_logger().info(text)
        self.pub_event.publish(String(data=text))

    def _set_phase(self, phase):
        self.phase, self.phase_t0 = phase, time.time()
        if phase == 'START':
            self.goal_hits = []         # nothing from before the (re)start counts
            self.armed = False          # e.g. 바로 시작 with the ball still lying in the goal
        self.pub_control.publish(String(data='WAIT' if phase in ('FINISHED', 'CORNER') else phase))

    def _robot_state(self, msg):
        self.robot_state, self.robot_state_t = msg.data, time.time()
        self._team_state('YELLOW', msg.data)

    def _team_state(self, team, state):
        self.states[team] = (state, time.time())
        if team == self.scorer and state.startswith('CELEBRATE'):
            self.celebration_seen = True

    def _goal_confirmed(self, now):
        """A goal only from repeated estimates behind the same goal line (>= goal_hits within goal_window),
        and not while the other robot sees the ball clearly elsewhere at the same time (a mis-estimate)."""
        win = self.get_parameter('goal_window').value
        self.goal_hits = [h for h in self.goal_hits if now - h[2] < win]
        for team in ('YELLOW', 'BLUE'):
            hits = [h for h in self.goal_hits if h[0] == team]
            if len(hits) < self.get_parameter('goal_hits').value:
                continue
            sources = {h[1] for h in hits}
            gx = (self.L / 2 + 0.05) * (1 if team == 'YELLOW' else -1)
            veto = False
            for src, (x, y, t) in self.ball_src.items():
                if src not in sources and now - t < 0.3 and math.hypot(x - gx, y) > 0.35:
                    veto = True                           # the other camera sees the ball well away from the goal
            if veto:
                self.goal_hits = [h for h in self.goal_hits if h[0] != team]
                self.get_logger().warning(f'{team} goal ignored: the other robot sees the ball elsewhere')
                continue
            return team
        return None

    def _ready(self, now):
        teams = ('YELLOW', 'BLUE') if self.duel else ('YELLOW',)
        return all(self.states[t][0] == 'READY' and now - self.states[t][1] < 2.0 for t in teams)

    def _celebration_over(self, now):
        elapsed = now - self.phase_t0
        if elapsed < self.get_parameter('goal_pause').value:
            return False
        st, t = self.states[self.scorer]
        if st == 'CELEBRATE_DONE' and now - t < 2.0:
            return True
        if elapsed > self.get_parameter('celebrate_timeout').value:
            return True
        return not self.celebration_seen and elapsed > self.get_parameter('goal_pause').value + 3.0

    def _ball(self, msg, source='YELLOW'):
        x, y, now = msg.point.x, msg.point.y, time.time()
        if abs(x) > self.L / 2 + 0.15 or abs(y) > self.W / 2 + 0.15:
            return                                        # beyond the walls: a bad estimate, ignore
        prev = self.ball_src.get(source)
        if prev is not None and now - prev[2] < 0.3 and math.hypot(x - prev[0], y - prev[1]) > 0.6:
            self.ball_src[source] = (x, y, now)           # a jump no ball makes in 0.3 s: wait for it to repeat
            return
        self.ball_src[source] = (x, y, now)
        # goal candidates: estimates inside a goal mouth, behind its line (counted, see _goal_confirmed)
        if abs(y) < self.G / 2 and self.L / 2 + 0.005 < abs(x) < self.L / 2 + 0.12:
            self.goal_hits.append(('YELLOW' if x > 0 else 'BLUE', source, now))
        self.ball = (x, y, now)
        ref = self.still_src.get(source)
        if ref is None or now - ref[3] > 0.6 or \
                math.hypot(x - ref[0], y - ref[1]) > self.get_parameter('ball_still_tol').value:
            self.still_src[source] = (x, y, now, now)       # (x, y, still since, last seen)
        else:
            self.still_src[source] = (ref[0], ref[1], ref[2], now)
        self.still_ref = self._still_ball(now)

    def _still_ball(self, now):
        """The camera estimate that has been lying still longest (and is fresh): (x, y, still since) or None."""
        best = None
        for x, y, since, seen in self.still_src.values():
            if now - seen < 0.6 and (best is None or since < best[2]):
                best = (x, y, since)
        return best

    def _amcl(self, msg):
        p = msg.pose.pose
        x, y = p.position.x, p.position.y
        hx, hy = self.L / 2 + self.strip - 0.06, self.W / 2 + self.strip - 0.06   # robot centre limits
        now = time.time()
        if abs(x) <= hx + 0.03 and abs(y) <= hy + 0.03:
            self.outside_since = None
            return
        if self.outside_since is None:
            self.outside_since = now
            return
        if now - self.outside_since < 1.0 or now - self.last_reset < 3.0:
            return
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x = max(-hx, min(hx, x))
        m.pose.pose.position.y = max(-hy, min(hy, y))
        m.pose.pose.orientation = p.orientation
        m.pose.covariance[0] = m.pose.covariance[7] = 0.01        # 10 cm: let AMCL find it again
        m.pose.covariance[35] = 0.05
        self.pub_initialpose.publish(m)
        self.last_reset, self.outside_since = now, None
        self._event(f'위치 추정이 벽 밖({x:.2f}, {y:.2f})으로 나가 경기장 안으로 재설정')

    def _control(self, msg):
        cmd = msg.data.strip().upper()
        if cmd == 'NEW_GAME':
            self.yellow = self.blue = 0
            self.goals = []
            self.game_t0 = time.time()
            self._event('새 경기 - 노랑이 자기 골대 앞으로 갑니다')
            self._prepare()
        elif cmd == 'RESET' and self.phase not in ('WAIT', 'FINISHED'):   # CORNER included
            self._event('원위치 - 노랑이 자기 골대 앞으로 갑니다')
            self._prepare()
        elif cmd == 'UNDO_GOAL':
            if not self.goals:
                self._event('취소할 골이 없습니다')
                return
            _, team, _, _, _ = self.goals.pop()
            if team == 'YELLOW':
                self.yellow = max(0, self.yellow - 1)
            else:
                self.blue = max(0, self.blue - 1)
            self._event(f'{"노랑" if team == "YELLOW" else "파랑"} 골 취소 -> {self.yellow} : {self.blue}')
            self._prepare()
        elif cmd == 'FORCE_START':
            if self.phase == 'FINISHED':
                self.yellow = self.blue = 0
                self.goals = []
            if not self.goals and self.phase in ('WAIT', 'FINISHED'):
                self.game_t0 = time.time()
            self.ball_at_mouth_t = None
            self.corner_since = None
            self._set_phase('START')
            self._event('바로 시작! - 지금 위치에서 노랑 출발')
        elif cmd == 'STOP':
            self._set_phase('WAIT')
            self._event('경기 중지')

    # ------------------------------------------------------------------ flow
    def _place_ball(self, x, y):
        self.still_ref = None
        self.still_src = {}
        z = self.get_parameter('ball_radius').value + 0.003
        if self.get_parameter('sim_backend').value == 'mujoco':       # mujoco_soccer: /sim/set_pose
            from geometry_msgs.msg import PoseStamped
            m = PoseStamped()
            m.header.frame_id = 'golf_ball'
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z
            m.pose.orientation.w = 1.0
            self.pub_sim_pose.publish(m)
            return
        import threading
        from pinky_soccer_sim.gz_utils import gz_set_pose
        threading.Thread(target=gz_set_pose, daemon=True,
                         args=(self.get_parameter('world').value, 'golf_ball', x, y, z, 0.0)).start()

    def _prepare(self):
        self.ball_placed = False
        self.asked_ball = False
        self.ball_at_mouth_t = None
        self._set_phase('PREPARE')

    def _goal(self, team, how):
        if team == 'YELLOW':
            self.yellow += 1
        else:
            self.blue += 1
        t = int(time.time() - self.game_t0)
        n = len(self.goals) + 1
        win = self.get_parameter('win_score').value
        mine = self.yellow if team == 'YELLOW' else self.blue
        if self.duel:
            kind = '결승골' if win > 0 and mine >= win else '선제골' if n == 1 else '골'
        else:
            kind = '결승골' if win > 0 and mine >= win else '선제골' if n == 1 else '추가골' if team == 'YELLOW' \
                else '자책골'
        self.goals.append((f'{t // 60:02d}:{t % 60:02d}', team, self.yellow, self.blue, kind))
        name = '노랑' if team == 'YELLOW' else '파랑'
        line = f'{self.goals[-1][0]}  {name} {kind}!  {self.yellow} : {self.blue}  ({how})'
        # celebration: the scorer goes to the middle of its own half and spins (GOAL lasts until it is done)
        self.scorer = team
        self.pub_goal.publish(String(data=team))
        self.celebration_seen = False
        if win > 0 and max(self.yellow, self.blue) >= win:
            self.pub_fx.publish(String(data='FINISHED'))
            self.finish_after_goal = f'{line}  -  경기 종료, {name} 승리!'
            self._set_phase('GOAL')
            self._event(f'{line}  -  세레모니 후 경기 종료')
        else:
            self.pub_fx.publish(String(data='GOAL'))
            self.finish_after_goal = None
            self._set_phase('GOAL')
            self._event(f'{line}  -  세레모니(제자리 3바퀴) 후 출발 위치로, 공을 가운데에 놓아 주세요')

    def tick(self):
        now = time.time()
        b = self.ball
        seen = b is not None and now - b[2] < 0.6
        if self.phase == 'GOAL' and self._celebration_over(now):
            if self.finish_after_goal:
                self._set_phase('FINISHED')
                self._event(self.finish_after_goal)
                self.finish_after_goal = None
            else:
                self._prepare()
        elif self.phase == 'CORNER' and now - self.phase_t0 > self.get_parameter('corner_pause').value:
            self._prepare()
        elif self.phase == 'PREPARE':
            ready = self._ready(now)
            if ready and self.get_parameter('place_ball_sim').value and not self.ball_placed:
                self.ball_placed = True
                self._place_ball(0.0, 0.0)
                self._event('(시뮬) 공을 가운데에 놓았습니다')
            ref = self._still_ball(now)
            if ref is not None:
                b = (ref[0], ref[1], now)                 # the steady camera's estimate of the ball
            centred = seen and math.hypot(b[0], b[1]) < self.get_parameter('centre_tol').value
            still = seen and ref is not None and now - ref[2] > self.get_parameter('ball_still_time').value
            if now - self.phase_t0 > 8.0 and now - self.wait_note_t > 8.0 and not (ready and centred and still):
                self.wait_note_t = now                    # say what the auto-start is still waiting for
                st = {t: self.states[t][0] if now - self.states[t][1] < 2.0 else '(신호 없음)'
                      for t in (('YELLOW', 'BLUE') if self.duel else ('YELLOW',))}
                why = [f'{"노랑" if t == "YELLOW" else "파랑"} {v}' for t, v in st.items() if v != 'READY']
                if not seen:
                    why.append('공 안 보임')
                elif not centred:
                    why.append(f'공이 가운데 아님 ({b[0]:+.2f}, {b[1]:+.2f})')
                elif not still:
                    why.append('공이 아직 흔들림')
                text = '자동 시작 대기: ' + ', '.join(why)
                if text != getattr(self, 'last_wait_text', ''):
                    self.last_wait_text = text
                    self._event(text)                     # shown in the match window
                else:
                    self.get_logger().info(text)
            if ready and centred and still:
                self.ball_at_mouth_t = None
                self.corner_since = None
                self._set_phase('START')
                self._event('경기 시작 - 가운데 공 (직전 득점 팀은 3초 뒤 출발)' if self.goals else
                            '경기 시작 - 가운데 공, 노랑 선공 (파랑은 3초 뒤 출발)')
            elif ready and not centred and still and self.asked_ball != 'off_centre':
                self.asked_ball = 'off_centre'
                self._event(f'공이 가운데가 아닙니다 ({b[0]:.2f}, {b[1]:.2f}) - 가운데에 놓으면 자동 시작 '
                            f'(그대로 하려면 바로 시작)')
            elif ready and not seen and self.asked_ball != 'no_ball':
                self.asked_ball = 'no_ball'
                self._event('준비 완료 - 공을 가운데에 놓아 주세요 (가만히 2초 있으면 자동 시작)')
        elif self.phase == 'START':
            if seen and abs(b[0]) < self.L / 2 - 0.05:
                self.armed = True
            if not self.armed:
                return
            if seen and self.get_parameter('place_ball_sim').value and abs(b[0]) > self.L / 2 + 0.01 \
                    and abs(b[1]) >= self.G / 2 and now - self.last_out > 2.0:
                self._place_ball(0.0, 0.0)                      # no random spots: back to the centre
                self.last_out = now
                self._event('골대 옆으로 나간 공 -> 중앙, 경기 계속')
                return
            scorer = self._goal_confirmed(now)
            if scorer is not None:
                self._goal(scorer, '공이 골라인 통과' if (scorer == 'YELLOW' or self.duel) else '자기 골대에 들어감')
                return
            if seen and abs(b[0]) > self.L / 2 + 0.01 and abs(b[1]) >= self.G / 2:
                # out over the goal line beside the goal: no stop, the robots play on - only the ball goes
                # back to the centre spot (a person puts it there)
                self.corner_since = None
                if now - self.last_out > 5.0:
                    self.last_out = now
                    self.pub_fx.publish(String(data='CORNER'))
                    self._event('공이 골라인 밖으로 나감 - 로봇은 그대로, 공만 가운데에 놓아 주세요')
                return
            if seen:
                margin = self.get_parameter('mouth_margin').value
                mouth = abs(b[1]) < self.G / 2 - 0.02
                if self.duel:
                    self.ball_at_mouth_t = None   # two robots: the other one often hides the ball - no guessing
                elif mouth and b[0] > self.L / 2 - margin:
                    self.ball_at_mouth_t, self.mouth_team = now, 'YELLOW'
                elif mouth and self.duel and b[0] < -self.L / 2 + margin:
                    self.ball_at_mouth_t, self.mouth_team = now, 'BLUE'
                else:
                    self.ball_at_mouth_t = None
                ref = self._still_ball(now)
                wz = self.get_parameter('wall_zone').value
                on_wall = ((self.L / 2 - abs(b[0]) < wz and abs(b[1]) > self.G / 2) or self.W / 2 - abs(b[1]) < wz)
                stuck = ref is not None and now - ref[2] > self.get_parameter('stuck_time').value
                if stuck and not on_wall and now - self.last_out > 10.0:
                    self.last_out = now
                    self.pub_fx.publish(String(data='CORNER'))
                    self._event('공이 20초 동안 안 움직임 (막힘) - 로봇은 그대로, 공만 가운데에 놓아 주세요')
                if on_wall and ref is not None and now - ref[2] > self.get_parameter('wall_stuck_time').value \
                        and now - self.last_out > 10.0:
                    self.last_out = now
                    self.pub_fx.publish(String(data='CORNER'))
                    self._event('공이 라인에 붙어 안 움직임 - 로봇은 그대로, 공만 가운데에 놓아 주세요')
                zone = self.get_parameter('corner_zone').value
                in_corner = abs(b[0]) > self.L / 2 + self.strip - zone and abs(b[1]) > self.W / 2 + self.strip - zone
                if not in_corner:
                    self.corner_since = None
                elif self.corner_since is None:
                    self.corner_since = now
                elif now - self.corner_since > self.get_parameter('corner_time').value:
                    self.corner_since = None
                    self.pub_fx.publish(String(data='CORNER'))
                    self._set_phase('CORNER')
                    self._event('코너! 로봇은 제자리로 돌아갑니다 - 공을 가운데에 다시 놓아 주세요')
            elif self.ball_at_mouth_t is not None and \
                    now - self.ball_at_mouth_t > self.get_parameter('lost_goal_time').value:
                self._goal(getattr(self, 'mouth_team', 'YELLOW'), '골 입구에서 사라짐')


def main():
    rclpy.init()
    node = RealReferee()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
