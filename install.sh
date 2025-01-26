#!/bin/bash

# Create ROS workspace
mkdir -p ros_space
cd ros_space

# Clone necessary repositories
git clone https://github.com/Aoxcis/uvlarm-25.git
git clone https://github.com/imt-mobisyst/pkg-interfaces.git
git clone https://github.com/imt-mobisyst/pkg-tsim



# Install Python 3.10 and ROS2-Iron
sudo apt install python3.10
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install ros-dev-tools
sudo apt update
sudo apt upgrade
sudo apt install ros-iron-desktop

# Build the workspace
colcon build
source install/setup.bash

# Install required Python packages
pip install -r o2p5/grp_pibot25/requirements.txt

# Install Gazebo and related packages
sudo apt install -y gazebo \
    gazebo-common \
    gazebo-plugin-base \
    libgazebo-dev \
    libgazebo11:amd64 \
    ros-iron-gazebo-dev \
    ros-iron-gazebo-msgs \
    ros-iron-gazebo-plugins \
    ros-iron-gazebo-ros \
    ros-iron-gazebo-ros-pkgs \
    ros-iron-turtlebot3-gazebo

# Install navigation packages
sudo apt install -y ros-iron-navigation2 \
    ros-iron-nav2-bringup

# Install ROS2-Iron Python packages
sudo apt install -y ros-iron-rclpy \
    ros-iron-sensor-msgs \
    ros-iron-std-msgs \
    ros-iron-cv-bridge

# Install camera package
sudo apt install -y libsdl2-2.0-0