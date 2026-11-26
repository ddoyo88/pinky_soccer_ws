#!/usr/bin/env python3
"""Relay the shared game topics between the two robots' ROS domains (yellow 37 <-> blue 38).

Each robot (and its PC-side stack) lives in its own ROS_DOMAIN_ID with its own namespaced topics, so the two
robots cannot disturb each other.  Only the match itself is shared: the referee / match window run in the
yellow domain, and this process copies their topics to the blue domain (and the blue robot's state back).

    ros2 run pinky_soccer_game domain_relay --from-domain 37 --to-domain 38

std_msgs/String only (all /game/* topics are strings).  One process, one rclpy context per domain.
"""
import argparse

import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

# referee / match UI (home domain) -> the other domain
FORWARD = ['/game/control', '/game/kickoff', '/game/phase', '/game/score', '/game/event', '/game/fx',
           '/game/goal', '/game/attack']
# the other domain -> home domain: (topic there, topic here, type)
#   sim duel: the blue stack is namespaced (/blue/...); real robots: plain names in their own domain
BACKWARD = [('/blue/game/state', '/blue/game/state', 'String'),
            ('/game/state', '/blue/game/state', 'String'),
            ('/ball/position_map', '/blue/ball/position_map', 'PointStamped')]


def make_node(domain, name):
    ctx = Context()
    rclpy.init(context=ctx, domain_id=domain)
    return Node(name, context=ctx), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-domain', type=int, default=37)
    ap.add_argument('--to-domain', type=int, default=38)
    args, _ = ap.parse_known_args()
    home, home_ctx = make_node(args.from_domain, 'game_relay_home')
    other, other_ctx = make_node(args.to_domain, 'game_relay_other')

    from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
    types = {'String': String, 'PointStamped': PointStamped, 'PoseWithCovarianceStamped': PoseWithCovarianceStamped}

    def link(src, dst, topic_src, topic_dst=None, kind='String'):
        pub = dst.create_publisher(types[kind], topic_dst or topic_src, 10)
        src.create_subscription(types[kind], topic_src, pub.publish, 10)

    for t in FORWARD:
        link(home, other, t)
    # sim referee: blue's AMCL reset after it lifted the blue robot out of a goal (sim_duel: namespaced /blue)
    link(home, other, '/blue/initialpose', kind='PoseWithCovarianceStamped')
    for t_src, t_dst, kind in BACKWARD:
        link(other, home, t_src, t_dst, kind)
    home.get_logger().info(f'relaying {FORWARD} domain {args.from_domain} -> {args.to_domain}, '
                           f'{BACKWARD} back')
    ex_home = MultiThreadedExecutor(context=home_ctx)
    ex_other = MultiThreadedExecutor(context=other_ctx)
    ex_home.add_node(home)
    ex_other.add_node(other)
    import threading
    threading.Thread(target=ex_other.spin, daemon=True).start()
    try:
        ex_home.spin()
    except KeyboardInterrupt:
        pass
    for n, c in ((home, home_ctx), (other, other_ctx)):
        n.destroy_node()
        if c.ok():
            rclpy.shutdown(context=c)


if __name__ == '__main__':
    main()
