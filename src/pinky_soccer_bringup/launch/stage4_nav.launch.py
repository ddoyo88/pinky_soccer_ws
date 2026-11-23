"""STAGE 4 - Nav2 (official gz_bringup_launch + our field params) with AMCL on the ideal map, or on your scanned map.
Use "2D Goal Pose" in RViz.  The blue Pinky (blue_mode:=patrol) is a moving obstacle to avoid.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from pinky_soccer_bringup.common import field_declarations, field_forward, include, include_any


def generate_launch_description():
    return LaunchDescription(field_declarations() + [
        DeclareLaunchArgument('with_blue', default_value='true'),
        DeclareLaunchArgument('map', default_value=''),
        DeclareLaunchArgument('blue_mode', default_value='patrol'),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': LaunchConfiguration('with_blue'),
                 'cam_tilt_deg': LaunchConfiguration('cam_tilt_deg')}),
        include('pinky_soccer_nav', 'nav.launch.py',
                {**field_forward(), 'map': LaunchConfiguration('map')}),
        include('pinky_soccer_game', 'opponent.launch.py',
                {**field_forward(), 'blue_mode': LaunchConfiguration('blue_mode')}),
        include_any('pinky_navigation', 'gz_nav2_view.launch.xml'),    # official RViz for Nav2
    ])
