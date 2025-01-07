import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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
        )
    ])