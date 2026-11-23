"""STAGE 1 - look at the official Pinky Pro model in RViz.  No simulator.
Move the wheels with the slider window.  Uses the official RViz config of pinky_description.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    tilt = LaunchConfiguration('cam_tilt_deg').perform(context)
    dribbler = LaunchConfiguration('dribbler').perform(context)
    xacro_file = os.path.join(get_package_share_directory('pinky_soccer_description'),
                              'urdf', 'pinky_soccer.urdf.xacro')
    urdf = xacro.process_file(xacro_file, mappings={
        'namespace': '', 'is_sim': 'false', 'cam_tilt_deg': tilt, 'dribbler': dribbler}).toxml()
    rviz_cfg = os.path.join(get_package_share_directory('pinky_description'), 'rviz', 'view_robot.rviz')
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher', output='screen',
             parameters=[{'ignore_timestamp': False, 'robot_description': urdf}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             name='joint_state_publisher', output='screen'),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', rviz_cfg], on_exit=Shutdown()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        DeclareLaunchArgument('dribbler', default_value='false'),
        OpaqueFunction(function=setup),
    ])
