"""Shared launch helpers for the stage launch files."""
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

FIELD_ARGS = [
    ('field_style', 'hardboard', 'hardboard = four-board training field | rcj = RoboCupJunior | box = simple '
                                 '(sizes in pinky_soccer_sim/config/field_<style>.yaml)'),
    ('field_length', 'auto', 'playing length along X [m] (goal lines at +-L/2); auto = from the field YAML'),
    ('field_width', 'auto', 'playing width along Y [m]; auto = from the field YAML'),
    ('goal_width', 'auto', 'goal opening [m]; auto = from the field YAML'),
]


def field_declarations():
    return [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in FIELD_ARGS]


def field_forward():
    return {n: LaunchConfiguration(n) for n, _, _ in FIELD_ARGS}


def include(pkg, launch_file, args=None):
    """Include a python launch file of <pkg>."""
    path = os.path.join(get_package_share_directory(pkg), 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                    launch_arguments=(args or {}).items())


def include_any(pkg, launch_file, args=None):
    """Include any launch file (.xml / .py / .yaml) of <pkg> - the official pinky_* packages use XML launches."""
    path = os.path.join(get_package_share_directory(pkg), 'launch', launch_file)
    return IncludeLaunchDescription(AnyLaunchDescriptionSource(path),
                                    launch_arguments=(args or {}).items())


def rviz(config_pkg, config_file, sim_time=True, condition=None):
    cfg = os.path.join(get_package_share_directory(config_pkg), 'rviz', config_file)
    return Node(package='rviz2', executable='rviz2', arguments=['-d', cfg], output='log',
                parameters=[{'use_sim_time': sim_time}], condition=condition)


def solo_striker_params(L, W, G, strip, vmax, ready_gap=0.05, search_w=0.9, turn_max=2.0, orbit_radius=0.32,
                        match=True, sim_time=False):
    """Striker settings for ONE yellow robot alone on the field (real_yellow / sim_yellow use the same)."""
    return {'use_sim_time': sim_time,
            'field_length': L, 'field_width': W, 'goal_width': G, 'outer_strip': strip,
            'match_control': match,
            'start_delay': 8.0,                     # Nav2 + AMCL come up first
            'use_opponent': False,                  # alone on the field: no phantom dodging
            'ready_offset': ready_gap + 0.075,      # after a goal: robot back ready_gap from its own goal line
            'search_w': search_w,                   # look-around (camera ~53 deg; image-time TF keeps the ball
                                                    # position right while turning; 2026-11-24: 0.6 -> 0.9)
            'w_max': turn_max,                      # 2026-11-24: 1.2 -> 2.0 rad/s (was sluggish)
            'orbit_radius': orbit_radius,           # wide loop round the ball when there is no room to shoot
            'wall_margin': 0.0,                     # may line up right next to the walls
            'wall_guard_margin': -0.05,             # dribble may run along a wall (pushing a wall ball)
            'direct_approach_range': 1.5,           # own steering instead of Nav2 (keeps off walls, fast turns)
            'v_approach': vmax,                     # 2026-11-24: always full speed (was 0.22 / 0.15)
            'v_dribble': vmax,
            'v_shoot': vmax,
            'ball_timeout': 3.0}                    # keep the last seen ball 3 s: circling it turns the camera away
