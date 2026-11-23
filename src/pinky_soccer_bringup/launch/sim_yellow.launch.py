"""SIMULATION of the real single-robot game: the yellow Pinky alone, exactly the real_yellow.launch.py setup.

    ros2 launch pinky_soccer_bringup sim_yellow.launch.py

Same striker settings as on the real robot (pinky_soccer_bringup.common.solo_striker_params: 0.28 m/s, slow
search spin, wide orbit round the ball, right up to the walls, back to 5 cm in front of its own goal after a
goal), the same camera-based referee (real_referee: goal / corner / centre-only auto start / 바로 시작) and the
same match window.  Only difference: nobody is there to put the ball back, so the referee places it on the
centre spot itself once the robot is READY (place_ball_sim).  LED / buzzer effects exist only on the real robot.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_bringup.common import field_declarations, field_forward, include, rviz, solo_striker_params
from pinky_soccer_sim.field import resolve_dims


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    vmax = float(v('max_speed'))
    actions = [
        include('pinky_soccer_sim', 'sim.launch.py',
                {**field_forward(), 'with_blue': 'false', 'headless': v('headless')}),
        include('pinky_soccer_nav', 'nav.launch.py', {**field_forward(), 'max_linear': str(vmax)}),
        include('pinky_soccer_vision', 'ball_detector.launch.py', {'backend': v('backend')}),
        Node(package='pinky_soccer_game', executable='striker', name='striker', output='screen',
             parameters=[solo_striker_params(L, W, G, strip, vmax, float(v('ready_gap')), float(v('search_w')),
                                             float(v('turn_max')), float(v('orbit_radius')), True,
                                             sim_time=True)]),
        Node(package='pinky_soccer_game', executable='real_referee', name='referee', output='screen',
             parameters=[{'use_sim_time': True, 'field_length': L, 'field_width': W, 'goal_width': G,
                          'outer_strip': strip, 'win_score': int(v('win_score')), 'place_ball_sim': True}]),
        Node(package='pinky_soccer_game', executable='match_ui', name='match_ui', output='screen',
             parameters=[{'win_score': int(v('win_score'))}]),
    ]
    if v('camera_view').lower() == 'true':
        actions.append(Node(package='rqt_image_view', executable='rqt_image_view', name='camera_view',
                            arguments=['/ball/detection_image'], output='log'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        *field_declarations(),
        DeclareLaunchArgument('backend', default_value='hsv', description='hsv (camera, like the real robot) | gt'),
        DeclareLaunchArgument('max_speed', default_value='0.28', description='m/s, the real Pinky limit'),
        DeclareLaunchArgument('ready_gap', default_value='0.05'),
        DeclareLaunchArgument('search_w', default_value='0.6'),
        DeclareLaunchArgument('turn_max', default_value='1.2'),
        DeclareLaunchArgument('orbit_radius', default_value='0.32'),
        DeclareLaunchArgument('win_score', default_value='5'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('camera_view', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        OpaqueFunction(function=launch_setup),
        rviz('pinky_soccer_nav', 'soccer.rviz', condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
