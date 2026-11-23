"""SLAM (slam_toolbox) = the OFFICIAL pinky_navigation/launch/gz_map_building.launch.xml with field-sized params.

Drive around slowly with   ros2 run teleop_twist_keyboard teleop_twist_keyboard
Save the map with          ros2 run nav2_map_server map_saver_cli -f ~/maps/scanned_field
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory('pinky_soccer_nav')
    nav_share = get_package_share_directory('pinky_navigation')
    return LaunchDescription([
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(nav_share, 'launch', 'gz_map_building.launch.xml')),
            launch_arguments={
                'slam_params_file': os.path.join(share, 'config', 'mapper_params.yaml'),
                'use_sim_time': 'True',
            }.items()),
    ])
