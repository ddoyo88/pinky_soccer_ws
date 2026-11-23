"""Gazebo soccer simulation: field + golf ball + yellow Pinky Pro (+ optional blue opponent) + ROS bridges.

Mirrors the official pinky_gz_sim/launch/launch_sim.launch.xml (server + GUI, robot_state_publisher,
joint_state_publisher, ros_gz_sim create, parameter_bridge, ros_gz_image image_bridge), but with the soccer
world, a spawn pose, and an optional second robot.

ros2 launch pinky_soccer_sim sim.launch.py                       # four-board field (config/field_hardboard.yaml)
ros2 launch pinky_soccer_sim sim.launch.py field_style:=box field_length:=2.4 field_width:=1.8 with_blue:=true
"""
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pinky_soccer_sim.field import FieldSpec, build_world_sdf
from pinky_soccer_sim.urdf_tools import build_urdf

OUT_DIR = '/tmp/pinky_soccer'


def _s(context, name):
    return LaunchConfiguration(name).perform(context)


def _f(context, name, default=None):
    v = _s(context, name)
    return float(v) if v not in ('', 'auto') else default


def launch_setup(context):
    os.makedirs(OUT_DIR, exist_ok=True)
    L, W, G = _f(context, 'field_length'), _f(context, 'field_width'), _f(context, 'goal_width')
    spec = FieldSpec(length=L, width=W, goal_width=G, wall_height=_f(context, 'wall_height'),
                     style=_s(context, 'field_style'), ball_damping=_f(context, 'ball_damping'))
    L, W, G = spec.length, spec.width, spec.goal_width          # 'auto' -> from config/field_<style>.yaml
    world_path = os.path.join(OUT_DIR, 'soccer.sdf')
    with open(world_path, 'w') as f:
        f.write(build_world_sdf(spec, os.path.join(OUT_DIR, 'rcj_meshes'), step=_f(context, 'physics_step')))

    tilt = _f(context, 'cam_tilt_deg')
    dribbler = _s(context, 'dribbler').lower() == 'true'
    bumper = _s(context, 'bumper').lower() == 'true'
    with_blue = _s(context, 'with_blue').lower() == 'true'
    headless = _s(context, 'headless').lower() == 'true'
    blue_sensors = _s(context, 'blue_sensors').lower() == 'true'
    imu_ekf = _s(context, 'imu_ekf').lower() == 'true'
    use_sim_time = {'use_sim_time': True}
    camera = (int(_f(context, 'cam_width')), int(_f(context, 'cam_height')), _f(context, 'cam_rate'))

    yellow_xml = build_urdf(namespace='', cam_tilt_deg=tilt, dribbler=dribbler, bumper=bumper,
                            team='yellow', camera=camera, odom_tf=not imu_ekf)
    yellow_pose = (_f(context, 'yellow_x', -L / 4.0), _f(context, 'yellow_y', 0.0), _f(context, 'yellow_yaw', 0.0))

    sim_share = get_package_share_directory('pinky_soccer_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    desc_share = get_package_share_directory('pinky_description')
    soccer_desc_share = get_package_share_directory('pinky_soccer_description')

    # gz must resolve package://pinky_description/meshes/... and package://pinky_soccer_description/meshes/...
    # (same trick as the official launch)
    res = os.pathsep.join(p for p in (os.path.join(desc_share, '..'), os.path.join(soccer_desc_share, '..'),
                                      os.environ.get('GZ_SIM_RESOURCE_PATH', '')) if p)
    actions = [SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', res)]

    # --- simulator: server (+ GUI), exactly like the official launch ---
    server_args = f'-r -s -v2 {"--headless-rendering " if headless else ""}{world_path}'
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': server_args, 'on_exit_shutdown': 'true'}.items()))
    # The GUI is started only after both robots are spawned (see the end of this function): a GUI that is
    # already running can miss the 'new entity' update on a busy CPU and then never shows the robots.
    gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2'}.items())

    # --- yellow robot: robot_state_publisher + joint_state_publisher (same parameters as upload_robot.launch.py) ---
    actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher',
                        name='robot_state_publisher', output='screen',
                        parameters=[{'ignore_timestamp': False, 'robot_description': yellow_xml,
                                     'frame_prefix': ''}, use_sim_time]))
    actions.append(Node(package='joint_state_publisher', executable='joint_state_publisher',
                        name='joint_state_publisher', output='screen',
                        parameters=[{'source_list': ['joint_states'], 'rate': 20.0}, use_sim_time]))

    spawns = [Node(package='ros_gz_sim', executable='create', output='screen',
                   arguments=['-world', spec.world_name, '-name', 'pinky_yellow', '-topic', '/robot_description',
                              '-x', str(yellow_pose[0]), '-y', str(yellow_pose[1]), '-z', '0.03',
                              '-Y', str(yellow_pose[2])],
                   parameters=[use_sim_time])]

    bridges = [os.path.join(sim_share, 'config', 'bridge_pinky.yaml'),
               os.path.join(sim_share, 'config', 'bridge_world.yaml')]

    # --- blue opponent: same official model, namespace "blue/" (sensors on blue/scan, blue/camera/...) ---
    if with_blue:
        blue_xml = build_urdf(namespace='blue/', cam_tilt_deg=tilt, dribbler=dribbler,
                              strip_sensors=not blue_sensors, bumper=bumper, team='blue', camera=camera)
        actions.append(Node(package='robot_state_publisher', executable='robot_state_publisher',
                            name='robot_state_publisher', namespace='blue', output='screen',
                            parameters=[{'ignore_timestamp': False, 'robot_description': blue_xml,
                                         'frame_prefix': 'blue/'}, use_sim_time]))
        spawns.append(Node(package='ros_gz_sim', executable='create', output='screen',
                           arguments=['-world', spec.world_name, '-name', 'pinky_blue',
                                      '-topic', '/blue/robot_description',
                                      '-x', str(_f(context, 'blue_x', L / 4.0)),
                                      '-y', str(_f(context, 'blue_y', 0.0)), '-z', '0.03',
                                      '-Y', str(_f(context, 'blue_yaw', math.pi))],
                           parameters=[use_sim_time]))
        bridges.append(os.path.join(sim_share, 'config', 'bridge_blue.yaml'))
        if blue_sensors:
            bridges.append(os.path.join(sim_share, 'config', 'bridge_blue_sensors.yaml'))

    if imu_ekf:
        bridges.append(os.path.join(sim_share, 'config', 'bridge_imu.yaml'))
        actions.append(Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                            output='screen',
                            parameters=[os.path.join(sim_share, 'config', 'ekf_yellow.yaml'), use_sim_time]))
    for i, cfg in enumerate(bridges):
        actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                            name=f'gz_bridge_{i}', parameters=[{'config_file': cfg}, use_sim_time]))
    # camera image: through image_bridge exactly as in the official launch
    actions.append(Node(package='ros_gz_image', executable='image_bridge', name='image_bridge',
                        output='screen', arguments=['/camera/image_raw'], parameters=[use_sim_time]))
    if with_blue and blue_sensors:
        actions.append(Node(package='ros_gz_image', executable='image_bridge', name='blue_image_bridge',
                            output='screen', arguments=['/blue/camera/image_raw'], parameters=[use_sim_time]))

    # give gz a moment to load the world before spawning
    actions.append(TimerAction(period=3.0, actions=spawns))
    if not headless:
        pending = {id(n) for n in spawns}

        def _spawned(event, _context, node_id):
            pending.discard(node_id)
            return [TimerAction(period=1.0, actions=[gui])] if not pending else None

        for n in spawns:
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=n, on_exit=lambda e, c, i=id(n): _spawned(e, c, i))))
    return actions


def generate_launch_description():
    decl = [
        DeclareLaunchArgument('field_style', default_value='hardboard',
                              description='hardboard | rcj | box  (config/field_<style>.yaml)'),
        DeclareLaunchArgument('field_length', default_value='auto', description='auto = from the field YAML'),
        DeclareLaunchArgument('field_width', default_value='auto'),
        DeclareLaunchArgument('goal_width', default_value='auto'),
        DeclareLaunchArgument('wall_height', default_value='auto'),
        DeclareLaunchArgument('with_blue', default_value='false'),
        DeclareLaunchArgument('dribbler', default_value='false'),
        DeclareLaunchArgument('bumper', default_value='true', description='bumper_v1 on the robot front'),
        DeclareLaunchArgument('cam_tilt_deg', default_value='20',
                              description='camera pitch [deg]; official default is 8'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('ball_damping', default_value='0.05',
                              description='ball rolling resistance [N s/m]: larger = stops sooner, 0 = never'),
        DeclareLaunchArgument('physics_step', default_value='0.004',
                              description='physics step [s]; smaller = more accurate but slower'),
        DeclareLaunchArgument('imu_ekf', default_value='true',
                              description='yellow odom -> base_footprint from wheels + gyro (EKF) instead of wheels'),
        DeclareLaunchArgument('blue_sensors', default_value='true',
                              description='give the blue robot its lidar + camera (for the sensor view)'),
        DeclareLaunchArgument('cam_width', default_value='640', description='real Pinky: 1280'),
        DeclareLaunchArgument('cam_height', default_value='360', description='real Pinky: 720'),
        DeclareLaunchArgument('cam_rate', default_value='15', description='real Pinky: 30 Hz'),
        DeclareLaunchArgument('yellow_x', default_value='auto'),
        DeclareLaunchArgument('yellow_y', default_value='auto'),
        DeclareLaunchArgument('yellow_yaw', default_value='auto'),
        DeclareLaunchArgument('blue_x', default_value='auto'),
        DeclareLaunchArgument('blue_y', default_value='auto'),
        DeclareLaunchArgument('blue_yaw', default_value='auto'),
    ]
    return LaunchDescription(decl + [OpaqueFunction(function=launch_setup)])
