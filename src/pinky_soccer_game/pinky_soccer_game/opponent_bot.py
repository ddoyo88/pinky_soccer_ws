#!/usr/bin/env python3
"""Scripted opponent (blue) driven from Gazebo ground truth.  It has no sensors of its own.

mode:=attacker plays like the yellow striker (default): gets behind the ball, dribbles it toward the yellow
               goal and shoots - same speeds as yellow (approach 0.45, dribble 0.25, shot 0.65 m/s)
mode:=keeper   shadows the ball along its own goal mouth
mode:=chaser   drives to the ball and pushes it toward the yellow goal (-X)
mode:=idle     stands still (pure obstacle)
mode:=patrol   drives back and forth across the field (moving obstacle for Nav2 testing)
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from pinky_soccer_sim.gz_utils import PoseTracker, wrap
from pinky_soccer_sim.field import default_dims

from pinky_soccer_game.avoidance import PrepareRouter, avoid_opponent, detour_target, goal_mouth_guard

# defaults = the default field (pinky_soccer_sim/config/field_hardboard.yaml); launch files pass the real values
FIELD_L, FIELD_W, FIELD_G, FIELD_STRIP = default_dims()


class OpponentBot(Node):
    def __init__(self):
        super().__init__('opponent_bot')
        self.declare_parameter('mode', 'attacker')
        self.declare_parameter('field_length', FIELD_L)
        self.declare_parameter('field_width', FIELD_W)
        self.declare_parameter('goal_width', FIELD_G)
        self.declare_parameter('own_goal_sign', 1.0)     # blue defends +X
        self.declare_parameter('max_speed', 0.45)        # same as yellow's Nav2 speed (desired_linear_vel)
        self.declare_parameter('kick_time', 0.8)         # s of full-speed push after touching on its kickoff
        self.declare_parameter('keeper_offset', 0.20)    # distance in front of own goal line
        # attacker (same numbers as the yellow striker)
        self.declare_parameter('v_dribble', 0.25)
        self.declare_parameter('v_shoot', 0.65)
        self.declare_parameter('k_lateral', 5.0)
        self.declare_parameter('behind_dist', 0.10)
        self.declare_parameter('shoot_range', 0.70)
        self.declare_parameter('match_control', False)
        self.declare_parameter('outer_strip', FIELD_STRIP)
        self.L = self.get_parameter('field_length').value
        self.W = self.get_parameter('field_width').value
        self.G = self.get_parameter('goal_width').value
        self.sgn = self.get_parameter('own_goal_sign').value
        self.vmax = self.get_parameter('max_speed').value
        self.tracker = PoseTracker()
        strip = self.get_parameter('outer_strip').value
        self.router = PrepareRouter(self.L / 2 - 0.12, self.W / 2 + strip - 0.09)   # x: out of the goals
        self.patrol_dir = 1.0
        self.create_subscription(TFMessage, '/gz/dynamic_pose', self.tracker.update, 10)
        self.match_control = self.get_parameter('match_control').value
        self.match_phase = 'WAIT' if self.match_control else 'PLAY'
        if self.match_control:
            from std_msgs.msg import String
            self.create_subscription(String, '/game/control', self._match_control, 10)
            self.create_subscription(String, '/game/kickoff', self._kickoff, 10)
        self.kickoff = 'OPEN'
        self.kick_until = 0.0
        self.att_state = 'APPROACH'
        self.shoot_until = 0.0
        self.stuck_ref = None          # (x, y, yaw, t) last pose where the robot was seen moving
        self.backoff_until = 0.0
        self.pub = self.create_publisher(Twist, '/blue/cmd_vel', 10)
        self.create_timer(0.05, self.tick)

    def _kickoff(self, msg):
        new = msg.data.strip().upper()
        if self.kickoff == 'BLUE' and new == 'OPEN':          # just touched it: follow through
            self.kick_until = time.time() + self.get_parameter('kick_time').value
        self.kickoff = new

    def _match_control(self, msg):
        if msg.data == 'PREPARE':
            self.router.reset()
        if msg.data in ('PREPARE', 'START', 'GOAL', 'WAIT'):
            self.match_phase = 'PREPARING' if msg.data == 'PREPARE' else msg.data

    def go_to(self, me, tx, ty, tol=0.03, face=None, vmax=None):
        dx, dy = tx - me[0], ty - me[1]
        dist = math.hypot(dx, dy)
        cmd = Twist()
        if dist < tol:
            if face is not None:
                # in place: turn to the requested heading only (no forward/backward flip, no creeping)
                cmd.angular.z = max(-3.0, min(3.0, 3.0 * wrap(face - me[3])))
            return cmd
        heading = math.atan2(dy, dx)
        err = wrap(heading - me[3])
        # drive forward or backward, whichever needs less rotation (keeper shuffles sideways by turning)
        if abs(err) > math.pi / 2:
            err = wrap(err + math.pi)
            direction = -1.0
        else:
            direction = 1.0
        cmd.angular.z = max(-4.0, min(4.0, 4.0 * err))
        v = self.vmax if vmax is None else vmax
        cmd.linear.x = direction * min(v, 2.5 * dist) * max(0.0, math.cos(err)) if abs(err) < 1.2 else 0.0
        return cmd

    def unstick(self, me, cmd):
        """Wedged against the other robot / a wall: commanded to move but not moving for 1 s -> back off."""
        now = time.time()
        if now < self.backoff_until:
            out = Twist()
            out.linear.x, out.angular.z = -0.25, 1.0
            return out
        moving_cmd = abs(cmd.linear.x) > 0.05 or abs(cmd.angular.z) > 0.5
        ref = self.stuck_ref
        if ref is None or not moving_cmd or math.hypot(me[0] - ref[0], me[1] - ref[1]) > 0.015 \
                or abs(wrap(me[3] - ref[2])) > 0.08:
            self.stuck_ref = (me[0], me[1], me[3], now)
        elif now - ref[3] > 1.0:
            self.backoff_until = now + 0.5
            self.stuck_ref = None
        return cmd

    def attack(self, me, ball, opp=None):
        """Yellow-striker-like attack toward the yellow goal (x = -sgn * L/2), all from ground truth."""
        now = time.time()
        aim = (-self.sgn * self.L / 2, 0.0)
        bx, by = ball[0], ball[1]
        ax, ay = aim[0] - bx, aim[1] - by
        dist_aim = math.hypot(ax, ay)
        if dist_aim < 0.03:
            return Twist()
        ux, uy = ax / dist_aim, ay / dist_aim
        line_ang = math.atan2(uy, ux)
        dx, dy = me[0] - bx, me[1] - by
        along = ux * dx + uy * dy                      # < 0: robot behind the ball
        lat = ux * dy - uy * dx                        # > 0: robot left of the ball->aim line
        cmd = Twist()
        if now < self.shoot_until:
            cmd.linear.x = self.get_parameter('v_shoot').value
            cmd.angular.z = max(-4.0, min(4.0, 3.0 * wrap(line_ang - me[3])))
            return cmd
        lat_limit = 0.12 if self.att_state == 'DRIBBLE' else 0.05
        if -0.35 < along < -0.03 and abs(lat) < lat_limit:
            self.att_state = 'DRIBBLE'
            heading = line_ang - math.atan(self.get_parameter('k_lateral').value * lat)
            err = wrap(heading - me[3])
            cmd.angular.z = max(-4.0, min(4.0, 3.0 * err))
            cmd.linear.x = self.get_parameter('v_dribble').value * max(0.0, math.cos(err)) ** 2 \
                if abs(err) < 1.0 else 0.0
            if dist_aim < self.get_parameter('shoot_range').value and abs(lat) < 0.03 \
                    and abs(wrap(line_ang - me[3])) < 0.17:
                self.shoot_until = now + 0.6
            return cmd
        self.att_state = 'APPROACH'
        bd = self.get_parameter('behind_dist').value
        tx, ty = bx - ux * bd, by - uy * bd
        if along > -0.06 or abs(lat) > 0.20:           # wrong side: go around the ball, not through it
            side = 1.0 if lat >= 0 else -1.0
            tx, ty = bx - ux * 0.20 - uy * side * 0.20, by - uy * 0.20 + ux * side * 0.20
        lim_x, lim_y = self.L / 2 + 0.05, self.W / 2 + 0.05   # stay on the field / outer strip
        tx, ty = max(-lim_x, min(lim_x, tx)), max(-lim_y, min(lim_y, ty))
        if math.hypot(tx - me[0], ty - me[1]) < 0.04:  # on the spot: face along the line
            cmd.angular.z = max(-4.0, min(4.0, 4.0 * wrap(line_ang - me[3])))
            return cmd
        if opp is not None:                            # yellow in the way -> pass beside it, not through it
            (tx, ty), _ = detour_target((me[0], me[1], me[3]), (tx, ty), opp, self.router)
            tx, ty = max(-lim_x, min(lim_x, tx)), max(-lim_y, min(lim_y, ty))
        return self.go_to(me, tx, ty, tol=0.0)

    def tick(self):
        me, ball = self.tracker.get('pinky_blue'), self.tracker.get('golf_ball')
        if me is None:
            return
        if self.match_control and self.match_phase in ('WAIT', 'GOAL'):
            self.pub.publish(Twist())
            return
        mode = self.get_parameter('mode').value
        cmd = Twist()
        if self.match_control and self.match_phase in ('PREPARING', 'READY'):
            tx = self.sgn * (self.L / 2 - 0.38)
            if math.hypot(me[0] - tx, me[1]) < 0.10 and abs(wrap(me[3] - math.pi)) < 0.25:
                self.match_phase = 'READY'
                cmd = Twist()
            else:
                # straight home, but round the yellow robot (wide detour after getting stuck on it)
                yellow = self.tracker.get('pinky_yellow')
                route = self.router.update((me[0], me[1], me[3]), (tx, 0.0),
                                           None if yellow is None else (yellow[0], yellow[1]), time.time(),
                                           moving_wanted=math.hypot(me[0] - tx, me[1]) > 0.03)
                if route[0] == 'backoff':
                    cmd = Twist()
                    cmd.linear.x = route[1]
                elif route[2]:
                    cmd = self.go_to(me, route[1][0], route[1][1], tol=0.0)
                else:
                    cmd = self.go_to(me, tx, 0.0, tol=0.03, face=math.pi)
                if yellow is not None and route[0] != 'backoff':
                    cmd.linear.x, cmd.angular.z, _ = avoid_opponent((me[0], me[1], me[3]), cmd.linear.x,
                                                                    cmd.angular.z, (yellow[0], yellow[1]),
                                                                    w_max=4.0)
        elif self.match_control and self.match_phase == 'START' and self.kickoff == 'YELLOW':
            cmd = Twist()                                  # yellow's kickoff: wait for its first touch
        elif self.match_control and self.match_phase == 'START' and ball is not None \
                and (self.kickoff == 'BLUE' or time.time() < self.kick_until):
            # own kickoff: drive through the ball toward the yellow goal (-sgn * X)
            cmd = self.go_to(me, ball[0] - self.sgn * 0.5, ball[1], tol=0.02)
        elif mode == 'attacker' and ball is not None:
            yellow = self.tracker.get('pinky_yellow')
            opp = None if yellow is None else (yellow[0], yellow[1])
            cmd = self.unstick(me, self.attack(me, ball, opp))
            # collision filter: slow down / steer away when yellow is in the way
            cmd.linear.x, cmd.angular.z, _ = avoid_opponent((me[0], me[1], me[3]), cmd.linear.x,
                                                            cmd.angular.z, opp, (ball[0], ball[1]), w_max=4.0)
        elif mode == 'keeper' and ball is not None:
            gx = self.sgn * (self.L / 2 - self.get_parameter('keeper_offset').value)
            gy = max(-self.G / 2 + 0.08, min(self.G / 2 - 0.08, ball[1]))
            cmd = self.go_to(me, gx, gy, tol=0.03, face=math.pi / 2)
            # if the ball is rolling into the box, charge it
            if abs(ball[0] - gx) < 0.35 and abs(ball[1]) < self.G / 2:
                cmd = self.go_to(me, ball[0], ball[1], tol=0.02)
        elif mode == 'chaser' and ball is not None:
            ux, uy = -self.sgn, 0.0                       # push toward yellow goal
            tx, ty = ball[0] - ux * 0.14, ball[1] - uy * 0.14
            behind = (me[0] - ball[0]) * ux + (me[1] - ball[1]) * uy < -0.02
            cmd = self.go_to(me, ball[0], ball[1], tol=0.02) if behind and abs(me[1] - ball[1]) < 0.06 \
                else self.go_to(me, tx, ty + (0.2 if me[1] > ball[1] else -0.2) * (0 if behind else 1))
        elif mode == 'patrol':
            ty = self.patrol_dir * (self.W / 2 - 0.25)
            if abs(me[1] - ty) < 0.05:
                self.patrol_dir *= -1.0
            cmd = self.go_to(me, 0.0, ty, tol=0.03)
        # every move (attack, kickoff push, backing off, getting ready): never into a goal mouth
        cmd.linear.x = goal_mouth_guard((me[0], me[1], me[3]), cmd.linear.x, self.L / 2, self.G)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = OpponentBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
