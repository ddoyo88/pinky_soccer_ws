"""STAGE 3 - scan the field with slam_toolbox (official gz_map_building launch + official map-building RViz).
Drive slowly around with teleop, then save the map:
     ros2 run nav2_map_server map_saver_cli -f ~/maps/scanned_field
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from pinky_soccer_bringup.common import field_declarations, field_forward, include, include_any


def generate_launch_description():
    return LaunchDescription(field_declarations() + [
        DeclareLaunchArgument('with_blue', default_value='false'),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': LaunchConfiguration('with_blue'),
                 'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg')}),
        include('pinky_soccer_nav', 'slam.launch.py'),
        include_any('pinky_navigation', 'gz_map_view.launch.xml'),     # official RViz for map building
    ])
