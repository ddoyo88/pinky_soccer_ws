"""Scripted blue opponent only (used by stage 4 to test obstacle avoidance)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_sim.field import resolve_dims


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    return [Node(package='pinky_soccer_game', executable='opponent_bot', name='opponent_bot', output='screen',
                 parameters=[{'use_sim_time': True, 'field_length': L, 'field_width': W, 'goal_width': G,
                              'outer_strip': strip,
                              'mode': v('blue_mode')}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('field_style', default_value='hardboard'),
        DeclareLaunchArgument('field_length', default_value='auto'),
        DeclareLaunchArgument('field_width', default_value='auto'),
        DeclareLaunchArgument('goal_width', default_value='auto'),
        DeclareLaunchArgument('blue_mode', default_value='patrol'),
        OpaqueFunction(function=launch_setup),
    ])
