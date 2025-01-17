# file: my_nav2_remap.launch.py

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):
    # 1) Read the user-supplied 'remap_cmd_vel' argument
    remap_cmd_vel = LaunchConfiguration('remap_cmd_vel').perform(context)

    # 2) Include the standard Nav2 launch
    #    Adjust the path if you installed Nav2 in a custom location or distro
    nav2_launch_file = "/opt/ros/humble/share/nav2_bringup/launch/navigation_launch.py"
    
    nav2_included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={'params_file': '/path/to/nav2_params.yaml'}.items(),
    )

    # 3) Example custom node that remaps /cmd_vel → whatever user sets via 'remap_cmd_vel'
    example_node = Node(
        package='grp_pibot25',
        executable='straight_nav2',
        name='straight_nav2',
        remappings=[('/cmd_vel', remap_cmd_vel)]
    )

    return [
        nav2_included,
        example_node
    ]

def generate_launch_description():
    # Define an argument the user can override in the YAML file
    remap_cmd_vel_arg = DeclareLaunchArgument(
        'remap_cmd_vel',
        default_value='/cmd_vel',
        description='Remap the robot base velocity command topic'
    )

    # We use OpaqueFunction to dynamically insert our nodes
    return LaunchDescription([
        remap_cmd_vel_arg,
        OpaqueFunction(function=launch_setup)
    ])
