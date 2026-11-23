"""STAGE 2 - Gazebo field + golf ball + official Pinky Pro.  Drive it with:
     ros2 run teleop_twist_keyboard teleop_twist_keyboard
Add the opponent body with with_blue:=true.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from pinky_soccer_bringup.common import field_declarations, field_forward, include, rviz


def generate_launch_description():
    return LaunchDescription(field_declarations() + [
        DeclareLaunchArgument('with_blue', default_value='false'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': LaunchConfiguration('with_blue'),
                 'headless': LaunchConfiguration('headless'),
                 'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg')}),
        rviz('pinky_soccer_nav', 'sim.rviz'),
    ])
