"""SIMULATION, 1:1 with TWO full robots laid out exactly like the real setup on one PC:

    yellow  ROS_DOMAIN_ID 37   topics /yellow/cmd_vel /yellow/scan /yellow/odom /yellow/camera/...  TF /yellow/tf
    blue    ROS_DOMAIN_ID 38   topics /blue/cmd_vel   /blue/scan   ...                                TF /blue/tf

Each robot = simulated hardware (what the real Pinky publishes: gz bridges, robot_state_publisher, EKF) + its own
PC-side stack (map/AMCL/Nav2, ball detector, striker), all inside its namespace and its domain.  Frames are plain
(odom, base_footprint, ...) as on the real robots - every robot has its own TF topic, so nothing collides.
Shared: referee + match window in the yellow domain; domain_relay copies /game/* to the blue domain.

    ros2 launch pinky_soccer_bringup sim_duel.launch.py            MuJoCo (default), camera ball detection
    ros2 launch pinky_soccer_bringup sim_duel.launch.py sim:=gz    the same in Gazebo
    ros2 launch pinky_soccer_bringup sim_duel.launch.py backend:=gt

MuJoCo: one process (pinky_soccer_sim mujoco_soccer) simulates the field, ball and both robots and publishes each
robot's topics in its own domain (odom + TF from the wheel encoders, so no EKF).  Gazebo: gz bridges per domain.

Look at one robot:  ROS_DOMAIN_ID=38 ros2 topic echo /blue/game/state
"""
import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_bringup.common import field_declarations, field_forward, include, solo_striker_params
from pinky_soccer_sim.field import FieldSpec, build_world_sdf, field_config, resolve_dims
from pinky_soccer_sim.urdf_tools import build_urdf

OUT_DIR = '/tmp/pinky_soccer'
TF_REMAP = [('/tf', 'tf'), ('/tf_static', 'tf_static')]          # -> /<ns>/tf inside the namespace
SIM = {'use_sim_time': True}


def _bridge_yaml(team, gz_prefix, with_world):
    """What a real Pinky publishes, under /<team>/ (+ /clock; + ground truth for the referee)."""
    def e(ros, gz, ros_t, gz_t, direction):
        return {'ros_topic_name': ros, 'gz_topic_name': gz, 'ros_type_name': ros_t, 'gz_type_name': gz_t,
                'direction': direction}
    p = gz_prefix
    entries = [
        e('/clock', 'clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock', 'GZ_TO_ROS'),
        e(f'/{team}/scan', f'{p}scan', 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan', 'GZ_TO_ROS'),
        e(f'/{team}/camera/camera_info', f'{p}camera/camera_info', 'sensor_msgs/msg/CameraInfo',
          'gz.msgs.CameraInfo', 'GZ_TO_ROS'),
        e(f'/{team}/cmd_vel', f'{p}cmd_vel', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'),
        e(f'/{team}/joint_states', f'{p}joint_states', 'sensor_msgs/msg/JointState', 'gz.msgs.Model', 'GZ_TO_ROS'),
        e(f'/{team}/odom', f'{p}odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry', 'GZ_TO_ROS'),
        e(f'/{team}/imu', f'{p}imu_raw', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU', 'GZ_TO_ROS'),
    ]
    if with_world:
        for model in ('golf_ball', 'pinky_yellow', 'pinky_blue'):
            entries.append(e('/gz/dynamic_pose', f'/model/{model}/pose', 'tf2_msgs/msg/TFMessage', 'gz.msgs.Pose_V',
                             'GZ_TO_ROS'))
    path = os.path.join(OUT_DIR, f'bridge_{team}.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(entries, f, sort_keys=False)
    return path


def _start_offset(v):
    """Start pose = the robot just in front of the white penalty line of its own goal, rear on the line
    (field YAML penalty_depth + 0.075 m from the goal line), or start_offset:=<m> from the goal line."""
    if v('start_offset') != 'line':
        return float(v('start_offset'))
    # robot rear on the line, the whole robot in front of it (base_footprint -> rear = 0.075 m)
    return float(field_config(v('field_style')).get('penalty_depth', 0.38)) + 0.075


def _robot(context, team, domain, gz_prefix, pose, L, W, G, strip, v, sim='gz'):
    """Simulated Pinky hardware + its PC-side stack, in namespace `team` and ROS domain `domain`."""
    sim_share = get_package_share_directory('pinky_soccer_sim')
    tilt = float(v('cam_tilt_deg'))
    camera = (640, 360, 15.0)
    # gz: the blue model needs its gz topics prefixed; MuJoCo: both robots use the plain URDF (real-robot names)
    xml = build_urdf(namespace=gz_prefix if sim == 'gz' else '', cam_tilt_deg=tilt, team=team, camera=camera,
                     odom_tf=False, plain_frames=True)
    with open(os.path.join(sim_share, 'config', 'ekf_yellow.yaml')) as f:
        ekf = yaml.safe_load(f)['ekf_filter_node']['ros__parameters']
    ekf.update({'odom0': 'odom', 'imu0': 'imu'})
    x, y, yaw = pose
    attack_x = L / 2 if team == 'yellow' else -L / 2
    attack_yaw = 0.0 if team == 'yellow' else math.pi
    striker = solo_striker_params(L, W, G, strip, float(v('max_speed')), 
                                  match=True, sim_time=True)
    striker['policy_path'] = v(f'policy_{team}')       # '' = rule-based striker
    striker.update({'team': team, 'goal_x': attack_x, 'attack_yaw': attack_yaw,
                    'ready_offset': _start_offset(v),           # on the white line in front of the own goal
                    # avoid_opponent (default): the other robot nearby is an obstacle to go round
                    'use_opponent': v('avoid_opponent').lower() == 'true',
                    'orbit_radius': 0.24,                    # tighter loop round the ball: back on it sooner
                    'search_w': float(v('search_w')), 'w_max': float(v('turn_max'))})
    ns = team
    actions = [
        SetEnvironmentVariable('ROS_DOMAIN_ID', str(domain)),
        # --- what the real robot publishes ---
        Node(package='robot_state_publisher', executable='robot_state_publisher', namespace=ns,
             name='robot_state_publisher', output='log', remappings=TF_REMAP,
             parameters=[{'robot_description': xml, 'frame_prefix': '', 'ignore_timestamp': False}, SIM]),
    ]
    if sim == 'gz':
        actions += [
            Node(package='joint_state_publisher', executable='joint_state_publisher', namespace=ns,
                 name='joint_state_publisher', output='log',
                 parameters=[{'source_list': ['joint_states'], 'rate': 20.0}, SIM]),
            Node(package='ros_gz_bridge', executable='parameter_bridge', name=f'gz_bridge_{team}', output='log',
                 parameters=[{'config_file': _bridge_yaml(team, gz_prefix, team == 'yellow')}, SIM]),
            Node(package='ros_gz_image', executable='image_bridge', name=f'image_bridge_{team}', output='log',
                 arguments=[f'{gz_prefix}camera/image_raw'],
                 remappings=[(f'{gz_prefix}camera/image_raw', f'/{team}/camera/image_raw')], parameters=[SIM]),
            Node(package='robot_localization', executable='ekf_node', namespace=ns, name='ekf_filter_node',
                 output='log', remappings=TF_REMAP, parameters=[ekf, SIM]),
            TimerAction(period=3.0, actions=[Node(
                package='ros_gz_sim', executable='create', output='log',
                arguments=['-world', 'soccer', '-name', f'pinky_{team}', '-string', xml,
                           '-x', str(x), '-y', str(y), '-z', '0.03', '-Y', str(yaw)])]),
        ]
    actions += [
        # --- PC-side stack of this robot ---
        include('pinky_soccer_nav', 'nav.launch.py',
                {**field_forward(), 'namespace': ns, 'max_linear': v('max_speed'),
                 'real_robot': 'true',             # lidar-weighted AMCL (pushing / spinning makes the wheels slip)
                 'initial_x': str(x), 'initial_y': str(y), 'initial_yaw': str(yaw)}),

    ]
    if not (team == 'blue' and v('drill').lower() == 'true'):        # drill: blue stands parked, no striker
        # (and no camera: its ball estimates - with a lost localisation - misled yellow through the shared ball)
        actions.append(Node(package='pinky_soccer_vision', executable='ball_detector', namespace=ns, name='ball_detector',
             output='log', remappings=TF_REMAP,
             parameters=[{'backend': v('backend'), 'model_path': '', 'process_every_n': 1,
                          'field_length': L, 'field_width': W, 'goal_width': G}, SIM]))
        actions.append(Node(package='pinky_soccer_game', executable='striker', namespace=ns, name='striker',
                            output='screen', remappings=TF_REMAP, parameters=[striker]))
    return GroupAction(scoped=True, actions=actions)


def launch_setup(context):
    v = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    os.makedirs(OUT_DIR, exist_ok=True)
    L, W, G, strip = resolve_dims(v('field_style'), v('field_length'), v('field_width'), v('goal_width'))
    spec = FieldSpec(length=L, width=W, goal_width=G, style=v('field_style'))
    world_path = os.path.join(OUT_DIR, 'soccer_duel.sdf')
    with open(world_path, 'w') as f:
        f.write(build_world_sdf(spec, os.path.join(OUT_DIR, 'rcj_meshes'), step=0.004))

    desc = get_package_share_directory('pinky_description')
    soccer_desc = get_package_share_directory('pinky_soccer_description')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    res = os.pathsep.join(p for p in (os.path.join(desc, '..'), os.path.join(soccer_desc, '..'),
                                      os.environ.get('GZ_SIM_RESOURCE_PATH', '')) if p)
    yd, bd = int(v('yellow_domain')), int(v('blue_domain'))
    headless = v('headless').lower() == 'true'
    sim = v('sim')
    ypose, bpose = (-L / 4, 0.0, 0.0), (L / 4, 0.0, math.pi)
    if sim == 'mujoco':
        sim_actions = [Node(
            package='pinky_soccer_sim', executable='mujoco_soccer', output='screen',   # no name: it has 2 nodes
            additional_env={'MUJOCO_GL': 'glfw'},
            arguments=(['--viewer'] if not headless else []) + [
                '--field-style', v('field_style'), '--yellow-domain', str(yd), '--blue-domain', str(bd),
                # '=' form: a pose like -0.415,0,0 would otherwise be read as an option
                '--yellow-pose=' + ','.join(map(str, ypose)), '--blue-pose=' + ','.join(map(str, bpose)),
                '--cam-tilt', v('cam_tilt_deg'), '--max-linear', str(float(v('max_speed')) + 0.02),
                '--ball-roll', v('ball_roll')] + (['--simple-robots'] if v('simple_robots') == 'true' else [])
                + (['--record', v('record')] if v('record') else []))]
    else:
        sim_actions = [
            SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', res),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
                launch_arguments={'gz_args': f'-r -s -v2 {"--headless-rendering " if headless else ""}{world_path}',
                                  'on_exit_shutdown': 'true'}.items())]
    actions = sim_actions + [
        _robot(context, 'yellow', yd, '', ypose, L, W, G, strip, v, sim),
        _robot(context, 'blue', bd, 'blue/', bpose, L, W, G, strip, v, sim),
        # --- the match: referee + window in the yellow domain, relayed to the blue domain ---
        GroupAction(scoped=True, actions=[
            SetEnvironmentVariable('ROS_DOMAIN_ID', str(yd)),
            Node(package='pinky_soccer_game', executable='referee', name='referee', output='screen',
                 parameters=[{'field_length': L, 'field_width': W, 'goal_width': G, 'outer_strip': strip,
                              'match_control': True, 'win_score': int(v('win_score')),
                              'auto_ball_reset': v('auto_ball_reset').lower() == 'true',
                              'drill': v('drill').lower() == 'true', 'drill_time': float(v('drill_time')),
                              'game_mode': v('game_mode'),
                              'ready_offset': _start_offset(v), 'sim_backend': sim,
                              'yellow_initialpose_topic': '/yellow/initialpose'}, SIM]),
            Node(package='pinky_soccer_game', executable='match_ui', name='match_ui', output='screen',
                 parameters=[{'win_score': int(v('win_score'))}]),
        ]),
        Node(package='pinky_soccer_game', executable='domain_relay', name='domain_relay', output='screen',
             arguments=['--from-domain', str(yd), '--to-domain', str(bd)]),
    ]
    if v('sensor_view').lower() == 'true':      # live camera (with detection box) + lidar + sensor values
        actions.append(Node(package='pinky_soccer_game', executable='sensor_view', output='log',      # no name: one node per domain
                            arguments=['--duel', '--yellow-domain', str(yd), '--blue-domain', str(bd)]))
    if sim == 'gz' and not headless:
        actions.append(TimerAction(period=9.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': '-g -v2'}.items())]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        *field_declarations(),
        DeclareLaunchArgument('sim', default_value='mujoco', description='mujoco | gz'),
        DeclareLaunchArgument('yellow_domain', default_value='37'),
        DeclareLaunchArgument('blue_domain', default_value='38'),
        DeclareLaunchArgument('backend', default_value='hsv', description='hsv (camera) | gt'),
        DeclareLaunchArgument('max_speed', default_value='0.28', description='m/s top speed (moving to places; ball play itself is slower, ball_play.py)'),
        DeclareLaunchArgument('start_offset', default_value='line',
                              description="start pose from the own goal line [m]; 'line' = on the penalty line"),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20'),
        DeclareLaunchArgument('win_score', default_value='5'),
        DeclareLaunchArgument('auto_ball_reset', default_value='false'),   # true: referee moves the ball itself (tests)
        DeclareLaunchArgument('record', default_value='',
                              description='video file (.mp4) recorded by the simulator itself, e.g. ~/pinky_rl/videos/match.mp4'),
        DeclareLaunchArgument('simple_robots', default_value='true'),     # true: viewer draws simple robot shapes (real time)   # true: referee moves the ball itself (tests)
        DeclareLaunchArgument('turn_max', default_value='1.0', description='rad/s max turn rate (rule-based striker)'),
        DeclareLaunchArgument('search_w', default_value='1.6', description='rad/s spin while looking for the ball'),
        DeclareLaunchArgument('policy_yellow', default_value='',
                              description='RL policy .npz (pinky_soccer_rl.export) for yellow; empty = rules'),
        DeclareLaunchArgument('policy_blue', default_value='',
                              description='RL policy .npz for blue; empty = rules'),
        DeclareLaunchArgument('avoid_opponent', default_value='true',
                              description='true: the other robot is an obstacle (slow down / go round it); '
                                          'false: contact allowed'),
        DeclareLaunchArgument('ball_roll', default_value='0.00035', description='MuJoCo ball rolling friction [m]'),
        DeclareLaunchArgument('headless', default_value='false'),
        # attack drill: yellow alone, the ball dropped at random spots, drill_time s to score each
        DeclareLaunchArgument('game_mode', default_value='attack_defense',
                              description='attack_defense (roles, 1-minute attacks) | free (both attack)'),
        DeclareLaunchArgument('drill', default_value='false'),
        DeclareLaunchArgument('drill_time', default_value='60'),
        DeclareLaunchArgument('sensor_view', default_value='true',
                              description='window: each robot camera + detection, lidar, sensor values'),
        OpaqueFunction(function=launch_setup),
    ])
