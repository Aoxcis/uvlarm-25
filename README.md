
# Requirements & Installation

This project integrates ROS2, Python, OpenCV, and Intel RealSense. To run it on another machine, the following dependencies are required:

## 1. ROS2 Installation (Iron)
Follow the official ROS installation instructions to install ROS2 Iron:
- [ROS2 Installation Guide (Iron)](https://docs.ros.org/en/iron/Installation.html)

After installing ROS2 Iron, create your ROS workspace and clone this repository into it.

```bash
mkdir -p ~/ros_space/src
cd ~/ros_space/src
git clone <repository-url>
cd ~/ros_space
colcon build
source install/setup.bash
```

## 2. Intel RealSense SDK 2.0
Install the necessary RealSense drivers and tools:
- Install `librealsense2-dev` and RealSense tools.
- Verify hardware detection using `realsense-viewer`:

```bash
sudo apt install librealsense2-dev
realsense-viewer
```

## 3. Python Dependencies
Ensure you have Python 3 installed. Then, install the following Python packages:
- Numpy:
  ```bash
  pip install numpy
  ```
- OpenCV:
  ```bash
  pip install opencv-python
  ```
- pyrealsense2:
  ```bash
  pip install pyrealsense2
  ```
- cv_bridge (ROS2 package):
  ```bash
  sudo apt install ros-iron-cv-bridge
  ```

## 4. ROS2 Python Dependencies
Install the necessary ROS2 dependencies:
- `rclpy` : ROS2 Python client library for writing ROS nodes.
- `sensor_msgs` : For image data types (`Image`) used in ROS topics.
- `std_msgs` : For standard message types like `String` for publishing detection messages.
- `cv_bridge` : ROS2 package for converting between OpenCV images and ROS image messages.

Install these dependencies with:

```bash
sudo apt install ros-iron-rclpy ros-iron-sensor-msgs ros-iron-std-msgs ros-iron-cv-bridge
```

## 5. Robot-Specific Message Drivers
Install drivers to interpret robot-specific messages (bumper, laser, etc.):

```bash
cd $ROS_WORKSPACE
git clone https://github.com/imt-mobisyst/pkg-interfaces.git
colcon build --base-path pkg-interfaces
source ./install/setup.bash
```

## 6. Clone Necessary Repositories
Clone the required repositories into your workspace and build them:

```bash
cd ~/ros_space
git clone https://github.com/imt-mobisyst/pkg-tsim
colcon build
source ./install/setup.bash
```

## 7. Compilation and Execution
- Clone this repository into your ROS2 workspace.
- From the root of the workspace, build the project:

```bash
colcon build
source install/setup.bash
```

- Launch a Python script, for example:

```bash
ros2 launch grp_pibot25 simulation_v1_launch.yaml
```



By following these steps, you should be able to successfully set up and run the project on your machine.