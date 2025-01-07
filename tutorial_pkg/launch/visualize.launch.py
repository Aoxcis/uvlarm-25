import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tbot_sim_launch_dir = get_package_share_directory('tbot_sim')
    challenge_1_launch = os.path.join(tbot_sim_launch_dir, 'launch', 'challenge-1.launch.py')

    return LaunchDescription([
        # Include the challenge-1 launch file from tbot_sim package
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(challenge_1_launch)
        ),
        # RViz for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(
                os.path.dirname(__file__), 'default.rviz')]
        ),
        # Additional sensors or visualization-related nodes
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'robot_base']
        ),
        # Basic move node from tutorial_pkg
        Node(
            package='tutorial_pkg',
            executable='basic_move',
            name='basic_move'
        )
    ])