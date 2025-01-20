#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Import to find packages in the ROS 2 workspace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1) Find the Nav2 package directory
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    # Path to Nav2's navigation_launch.py within nav2_bringup
    nav2_launch_path = os.path.join(
        nav2_bringup_dir,
        'launch',
        'navigation_launch.py'
    )

    # 2) Find your custom package directory (e.g., 'grp_pibot25')
    grp_pibot25_dir = get_package_share_directory('grp_pibot25')
    # Path to your config file in grp_pibot25/config/nav2_params.yaml
    nav2_params_file = os.path.join(
        grp_pibot25_dir,
        'config',
        'nav2_params.yaml'
    )

    # 3) Include Nav2 with your custom params file
    nav2_included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_path),
        launch_arguments={
            'params_file': nav2_params_file,
            'use_sim_time': 'false'
        }.items()
    )

    return LaunchDescription([
        nav2_included,
        # Add other nodes or includes here as needed
    ])
