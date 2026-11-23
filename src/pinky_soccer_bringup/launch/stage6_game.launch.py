"""STAGE 6 - full 1:1 match: sim + Nav2/AMCL + ball detector + striker + referee + scripted opponent.

ros2 launch pinky_soccer_bringup stage6_game.launch.py                       # HSV vision, keeper opponent
ros2 launch pinky_soccer_bringup stage6_game.launch.py backend:=gt           # debug strategy with ground truth
ros2 launch pinky_soccer_bringup stage6_game.launch.py backend:=yolo model_path:=$HOME/pinky_soccer_runs/ball.pt
ros2 launch pinky_soccer_bringup stage6_game.launch.py field_length:=3.0 field_width:=2.0 goal_width:=0.7
ros2 launch pinky_soccer_bringup stage6_game.launch.py map:=$HOME/maps/scanned_field.yaml goal_x:=1.8 goal_y:=0.0
      (scanned map: origin = robot start pose (-L/4,0) -> goal centre is at x = L/4 + L/2 = 1.8 for L=2.4)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from pinky_soccer_bringup.common import field_declarations, field_forward, include, rviz


def generate_launch_description():
    return LaunchDescription(field_declarations() + [
        DeclareLaunchArgument('backend', default_value='hsv'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        DeclareLaunchArgument('blue_mode', default_value='attacker'),
        DeclareLaunchArgument('map', default_value='', description='empty = ideal generated field map'),
        DeclareLaunchArgument('goal_x', default_value='nan'),
        DeclareLaunchArgument('goal_y', default_value='0.0'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true', description='false saves CPU on small machines'),
        DeclareLaunchArgument('sensor_view', default_value='true',
                              description='window with each robot\'s camera + lidar'),
        DeclareLaunchArgument('match_ui', default_value='true',
                              description='match window (score, 경기 시작 / 원위치 / 직전 골 취소)'),
        DeclareLaunchArgument('win_score', default_value='5', description='first to this many goals wins'),
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': 'true', 'headless': LaunchConfiguration('headless'),
                 'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg')}),
        include('pinky_soccer_nav', 'nav.launch.py', {**field_forward(), 'map': LaunchConfiguration('map')}),
        include('pinky_soccer_vision', 'ball_detector.launch.py',
                {'backend': LaunchConfiguration('backend'),
                 'model_path': LaunchConfiguration('model_path')}),
        include('pinky_soccer_game', 'game.launch.py',
                {**field_forward(), 'blue_mode': LaunchConfiguration('blue_mode'),
                 'goal_x': LaunchConfiguration('goal_x'), 'goal_y': LaunchConfiguration('goal_y'),
                 'match_ui': LaunchConfiguration('match_ui'), 'win_score': LaunchConfiguration('win_score'),
                 'sensor_view': LaunchConfiguration('sensor_view')}),
        rviz('pinky_soccer_nav', 'soccer.rviz', condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
