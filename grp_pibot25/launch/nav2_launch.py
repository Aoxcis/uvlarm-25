#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Path to Nav2's navigation_launch.py
    nav2_launch_path = os.path.join(
        '/opt/ros/iron/share/nav2_bringup',  # or your distro
        'launch',
        'navigation_launch.py'
    )

    # Include Nav2 with the custom params file
    nav2_included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_path),
        launch_arguments={
            'params_file': '/o2p5/grp_pibot25/config/nav2_params.yaml'
        }.items()
    )

    return LaunchDescription([
        nav2_included,
        # Add other nodes or includes here
    ])
