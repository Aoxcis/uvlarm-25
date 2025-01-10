import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Realsense(Node):
    def __init__(self, fps=60):
        super().__init__('realsense_node')  # Node name
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bridge = CvBridge()

        # Configure the streams
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)

        # Start the pipeline
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except Exception as e:
            self.get_logger().error(f"Error starting the pipeline: {e}")
            exit(1)

        # Create image publishers
        self.color_publisher = self.create_publisher(Image, 'camera/color', 10)
        self.depth_publisher = self.create_publisher(Image, 'camera/depth', 10)

    def read_imgs(self):
        # Capture frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("Failed to get frames from camera.")
            return None, None

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap to depth image (optional)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Debugging - Show the images using OpenCV (for testing purposes)
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(1)

        return color_image, depth_colormap

    def publish_imgs(self):
        color_image, depth_image = self.read_imgs()

        if color_image is not None:
            # Convert color image to ROS message
            msg_color = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            msg_color.header.stamp = self.get_clock().now().to_msg()
            msg_color.header.frame_id = "camera_color_frame"
            self.color_publisher.publish(msg_color)

        if depth_image is not None:
            # Convert depth image to ROS message
            msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "bgr8")
            msg_depth.header.stamp = self.get_clock().now().to_msg()
            msg_depth.header.frame_id = "camera_depth_frame"
            self.depth_publisher.publish(msg_depth)

def process_img(args=None):
    rclpy.init(args=args)
    rs_node = Realsense()

    while rclpy.ok():
        rs_node.publish_imgs()
        rs_node.get_logger().info("Publishing images...")
        rclpy.spin_once(rs_node, timeout_sec=0.001)  # Non-blocking spin

    # Stop streaming and shutdown node
    rs_node.pipeline.stop()
    rs_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    process_img()
