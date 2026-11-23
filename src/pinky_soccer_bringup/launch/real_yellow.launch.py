"""REAL ROBOT - the yellow Pinky alone on the real field: find the orange ball, push it into the +X goal.

On the robot (ssh pinky@192.168.4.1, same ROS_DOMAIN_ID as the PC):
    ros2 launch pinky_bringup bringup_robot.launch.xml          # wheels, odom, lidar, /tf (official)
    python3 ~/pi_camera_publisher.py                            # camera -> /camera/image_raw (this package)
On the PC:
    ros2 launch pinky_soccer_bringup real_yellow.launch.py

Place the robot at the yellow start pose before launching: x = -L/2 + 0.38 (0.38 m in front of the -X goal
line, on the centre line), facing the +X goal.  AMCL starts there.  The PC runs map + AMCL + Nav2 + ball
detector + striker, all on wall-clock time.  No referee / opponent (they need the simulator's ground truth).

!! The simulator publishes /cmd_vel too.  Never run the simulation on the same ROS_DOMAIN_ID while the real
   robot is on: stop the sim, or give the robot + this terminal their own domain (export ROS_DOMAIN_ID=...).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_bringup.common import field_declarations, field_forward, include, rviz, solo_striker_params
from pinky_soccer_sim.field import resolve_dims


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    vmax = float(v('max_speed'))
    start_x = -L / 2 + 0.38
    match = v('match_ui').lower() == 'true'
    actions = []
    domain = os.environ.get('ROS_DOMAIN_ID', '')
    if domain in ('', '0'):
        actions.append(LogInfo(msg='[real_yellow] ROS_DOMAIN_ID is 0 (default) - make sure NO simulation is '
                                   'running on this network/domain, its /cmd_vel would drive the real robot'))
    actions += [
        include('pinky_soccer_nav', 'nav.launch.py',
                {**field_forward(), 'use_sim_time': 'false', 'max_linear': str(vmax), 'real_robot': 'true',
                 'initial_x': str(start_x), 'initial_y': '0.0', 'initial_yaw': '0.0', 'map': v('map')}),
        include('pinky_soccer_vision', 'ball_detector.launch.py',
                {'backend': 'hsv', 'use_sim_time': 'false', 'compressed': 'true',
                 'cam_pitch_deg': v('cam_pitch_deg'), 'process_every_n': '1'}),
        Node(package='pinky_soccer_game', executable='striker', name='striker', output='screen',
             parameters=[solo_striker_params(L, W, G, strip, vmax, float(v('ready_gap')), float(v('search_w')),
                                             float(v('turn_max')), float(v('orbit_radius')), match)]),
    ]
    if match:   # real referee (camera ball) + match window: goals counted, robot back to its own goal after each
        actions += [
            Node(package='pinky_soccer_game', executable='real_referee', name='referee', output='screen',
                 parameters=[{'field_length': L, 'field_width': W, 'goal_width': G, 'outer_strip': strip,
                              'win_score': int(v('win_score'))}]),
            Node(package='pinky_soccer_game', executable='match_ui', name='match_ui', output='screen',
                 parameters=[{'win_score': int(v('win_score'))}]),
        ]
    if v('camera_view').lower() == 'true':   # what the robot sees + the detection box
        actions.append(Node(package='rqt_image_view', executable='rqt_image_view', name='camera_view',
                            arguments=['/ball/detection_image'], output='log'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        *field_declarations(),
        DeclareLaunchArgument('map', default_value='', description='empty = ideal field map from the YAML'),
        DeclareLaunchArgument('max_speed', default_value='0.28',
                              description='m/s; the Pinky wheels top out at 100 rpm = 0.28 m/s'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('match_ui', default_value='true',
                              description='match window + real referee (goals from the camera); false = just play'),
        DeclareLaunchArgument('win_score', default_value='5'),
        DeclareLaunchArgument('ready_gap', default_value='0.05',
                              description='after a goal: robot back this far in front of its own goal line [m]'),
        DeclareLaunchArgument('orbit_radius', default_value='0.32',
                              description='m, wide loop round the ball when there is no room to shoot'),
        DeclareLaunchArgument('search_w', default_value='0.6', description='rad/s spin while searching'),
        DeclareLaunchArgument('turn_max', default_value='1.2', description='rad/s max turn rate'),
        DeclareLaunchArgument('camera_view', default_value='true', description='camera + detection window'),
        DeclareLaunchArgument('cam_pitch_deg', default_value='6.0',
                              description='real camera pitch down not in the robot model (measured 6 deg)'),
        OpaqueFunction(function=launch_setup),
        rviz('pinky_soccer_nav', 'soccer.rviz', sim_time=False, condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
