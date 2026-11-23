from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    s = lambda n: ParameterValue(LaunchConfiguration(n), value_type=str)  # noqa: E731
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='hsv', description='hsv | yolo | gt'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='true', description='false on the real robot'),
        DeclareLaunchArgument('compressed', default_value='false', description='true: JPEG camera (real robot)'),
        DeclareLaunchArgument('cam_pitch_deg', default_value='0.0', description='extra camera pitch down [deg]'),
        DeclareLaunchArgument('process_every_n', default_value='2', description='1 = every camera frame'),
        Node(package='pinky_soccer_vision', executable='ball_detector', name='ball_detector',
             output='screen',
             parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                          'backend': s('backend'), 'model_path': s('model_path'),
                          'compressed': ParameterValue(LaunchConfiguration('compressed'), value_type=bool),
                          'cam_pitch_deg': ParameterValue(LaunchConfiguration('cam_pitch_deg'), value_type=float),
                          'process_every_n': ParameterValue(LaunchConfiguration('process_every_n'), value_type=int)}]),
    ])
