
# Requirements & Installation

This project integrates ROS2, Python, OpenCV, and Intel RealSense. To run it on another machine, the following dependencies are required:

## 1. ROS2 Installation (Iron)
Follow the official ROS installation instructions to install ROS2 Iron:
- [ROS2 Installation Guide (Iron)](https://docs.ros.org/en/iron/Installation.html)

After installing ROS2 Iron, create your ROS workspace and clone this repository into it.

```bash
mkdir -p ~/ros_space
cd ~/ros_space
git clone https://github.com/Aoxcis/uvlarm-25.git
cd ~/ros_space
colcon build
source install/setup.bash
```

## 2. Python Dependencies
Ensure you have Python 3 installed. Then, install the following Python packages:
- Numpy:
  ```bash
  pip install -r requirements.txt
  ```
- cv_bridge (ROS2 package):
  ```bash
  sudo apt install ros-iron-cv-bridge
  ```

## 3. ROS2 Python Dependencies
Install the necessary ROS2 dependencies:
- `rclpy` : ROS2 Python client library for writing ROS nodes.
- `sensor_msgs` : For image data types (`Image`) used in ROS topics.
- `std_msgs` : For standard message types like `String` for publishing detection messages.
- `cv_bridge` : ROS2 package for converting between OpenCV images and ROS image messages.

Install these dependencies with:

```bash
sudo apt install ros-iron-rclpy ros-iron-sensor-msgs ros-iron-std-msgs ros-iron-cv-bridge
```

## 4. Robot-Specific Message Drivers
Install drivers to interpret robot-specific messages (bumper, laser, etc.):

```bash
cd $ROS_WORKSPACE
git clone https://github.com/imt-mobisyst/pkg-interfaces.git
colcon build --base-path pkg-interfaces
source ./install/setup.bash
```

## 5. Clone Necessary Repositories
Clone the required repositories into your workspace and build them:

```bash
cd $ROS_WORKSPACE
git clone https://github.com/imt-mobisyst/pkg-tsim
colcon build
source ./install/setup.bash
```

## 6. Gazebo Installation
If you plan to use Gazebo for simulation, you need to install the following Gazebo-related packages:

- **Gazebo**:
  ```bash
  sudo apt install gazebo=11.10.2+dfsg-1
  sudo apt install gazebo-common=11.10.2+dfsg-1
  sudo apt install gazebo-plugin-base=11.10.2+dfsg-1
  sudo apt install libgazebo-dev=11.10.2+dfsg-1
  sudo apt install libgazebo11:amd64=11.10.2+dfsg-1
  ```

- **ROS2 Gazebo Packages**:
  ```bash
  sudo apt install ros-iron-gazebo-dev=3.7.0-3jammy.20230622.191804
  sudo apt install ros-iron-gazebo-msgs=3.7.0-3jammy.20231117.090251
  sudo apt install ros-iron-gazebo-plugins=3.7.0-3jammy.20231117.111548
  sudo apt install ros-iron-gazebo-ros=3.7.0-3jammy.20231117.104944
  sudo apt install ros-iron-gazebo-ros-pkgs=3.7.0-3jammy.20231117.114324
  sudo apt install ros-iron-turtlebot3-gazebo
  ```

These packages are necessary for integrating Gazebo with ROS2, and for using Gazebo for robot simulation in the project.

## 7. Compilation and Execution
- From the root of the workspace, build the project:

```bash
colcon build
source install/setup.bash
```

- Launch a yaml file, for example:

```bash
ros2 launch grp_pibot25 simulation_v1_launch.yaml
```

---

By following these steps, you should be able to successfully set up and run the project on your machine.
