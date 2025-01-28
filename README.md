
# Introduction

This project's purpose is to make a Kobuki robot capable of autonomously exploring a map with intelligent (non-random) movement and using an Intel RealSense camera to detect objects on the map and mark their locations.
Video presentation: https://drive.google.com/file/d/1jdSDWwWQXPewI4sk96ihXOMziDcvPdg5/view?usp=sharing

## Challenges

### First Challenge

Demonstrate the robot's ability to navigate a cluttered environment and provide a live view of its surroundings.

### Second Challenge

Demonstrate the robot's ability to navigate a cluttered environment while locating specific objects. This involves building a map of the environment.

# Installation
<!-- 
## Automatic Installation

To automatically install all the necessary dependencies and clone the required repositories, run the provided `install.sh` script:
```bash
chmod +x install.sh
./install.sh
```

## Manual Installation -->
## Prerequisites

This project requires:

- **ROS2 Iron**
- **Python 3.10**
- **Gazebo**
- **Nav2**
- **OpenCV**
- **YOLOv5**
- **Intel RealSense**

Ensure your system meets these requirements before proceeding.

## Setting Up the Workspace

1. Create your ROS workspace and clone the required repositories:
   ```bash
   mkdir -p ros_space
   cd ros_space
   git clone https://github.com/Aoxcis/uvlarm-25.git
   git clone https://github.com/imt-mobisyst/pkg-interfaces.git
   git clone https://github.com/imt-mobisyst/pkg-tsim.git
   ```

## Installing Dependencies

### Python 3.10

Install Python 3.10:

```bash
sudo apt install python3.10
```

### ROS2 Iron

Install ROS2 Iron:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install ros-dev-tools
sudo apt update && sudo apt upgrade
sudo apt install ros-iron-desktop
colcon build
source install/setup.bash
```

### Python Packages

Install the required Python packages:

```bash
pip install -r uvlarm-25/grp_pibot25/requirements.txt
```

### Simulation Packages

Install the necessary packages for the simulation:

```bash
sudo apt install gazebo gazebo-common gazebo-plugin-base libgazebo-dev libgazebo11:amd64
sudo apt install ros-iron-gazebo-dev ros-iron-gazebo-msgs ros-iron-gazebo-plugins ros-iron-gazebo-ros ros-iron-gazebo-ros-pkgs ros-iron-turtlebot3-gazebo
```

### Navigation Packages

Install the navigation packages:

```bash
sudo apt install ros-iron-navigation2 ros-iron-nav2-bringup
```

### ROS2 Python Packages

Install the ROS2 Python packages:

```bash
sudo apt install ros-iron-rclpy ros-iron-sensor-msgs ros-iron-std-msgs ros-iron-cv-bridge
```

### Camera Package

Install the package for camera support:

```bash
sudo apt install libsdl2-2.0-0
```

# Launching the Code

### First Challenge

To launch the code for the first challenge, navigate to the `ros_space` directory and run:

For simulation:

```bash
colcon build
source ./install/setup.bash
ros2 launch grp_pibot25 simulation_v1_launch.yaml
```

With a real robot:

```bash
ros2 launch grp_pibot25 tbot_v1_launch.yaml
```

### Second Challenge

To launch the code for the second challenge, navigate to the `ros_space` directory and run:

For simulation:

```bash
colcon build
source ./install/setup.bash
ros2 launch grp_pibot25 simulation_v2.0_launch.yaml
```

With a real robot:

```bash
ros2 launch grp_pibot25 tbot_v2_launch.yaml
```