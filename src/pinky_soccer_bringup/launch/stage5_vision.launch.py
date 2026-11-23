"""STAGE 5 - ball perception.  backend:=hsv (default) | yolo model_path:=... | gt
View the overlay:  ros2 run rqt_image_view rqt_image_view /ball/detection_image
Dataset for YOLO training (sim must be running with with_blue:=true):
     ros2 run pinky_soccer_vision collect_dataset --out ~/datasets/golfball --num 1500
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from pinky_soccer_bringup.common import field_declarations, field_forward, include, rviz


def generate_launch_description():
    return LaunchDescription(field_declarations() + [
        DeclareLaunchArgument('with_blue', default_value='true'),
        DeclareLaunchArgument('backend', default_value='hsv'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        DeclareLaunchArgument('map', default_value=''),
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': LaunchConfiguration('with_blue'),
                 'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg')}),
        include('pinky_soccer_nav', 'nav.launch.py', {**field_forward(), 'map': LaunchConfiguration('map')}),
        include('pinky_soccer_vision', 'ball_detector.launch.py',
                {'backend': LaunchConfiguration('backend'),
                 'model_path': LaunchConfiguration('model_path')}),
        rviz('pinky_soccer_nav', 'soccer.rviz'),
    ])
