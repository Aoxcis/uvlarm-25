#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
import pyrealsense2 as rs


# Realsense Node:
class Realsense(Node):
    def __init__(self, fps=60):
        super().__init__('realsense')
        self.fps = fps
        self.publisher_ = self.create_publisher(String, 'realsense', 10)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
        self.count = 0
        self.refTime = time.process_time()
        self.get_logger().info("-")
        self.isOk = True
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
        if not self.found_rgb:
            self.get_logger().error("Depth camera required !!!")
            exit(0)

        self.bridge = CvBridge()

        # Start streaming
        self.pipeline.start(self.config)

    def read_imgs(self):
        # Wait for a coherent tuple of frames: depth, color and accel
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.first(rs.stream.color)
        depth_frame = frames.first(rs.stream.depth)

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

    def publish_imgs(self):
        # Apply color map on depth image from RealSense (converted to 8-bit per pixel)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)

        msg_depth = self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
        msg_image = self.bridge.cv2_to_imgmsg(self.color_image, "bgr8")

        msg_depth.header.stamp = msg_image.header.stamp
        msg_depth.header.frame_id = "depth"

        # Publish the messages
        self.publisher_.publish(msg_depth)
        self.publisher_.publish(msg_image)
        self.count += 1

    def timer_callback(self):
        # Read and publish images at the desired FPS
        if self.isOk:
            self.read_imgs()
            self.publish_imgs()


# Node processes:
def process_img(args=None):
    rclpy.init(args=args)
    rsNode = Realsense()
    while rsNode.isOk:
        rclpy.spin_once(rsNode, timeout_sec=0.001)
    
    # Stop streaming
    rsNode.get_logger().info("Ending...")
    rsNode.pipeline.stop()

    # Clean end
    rsNode.destroy_node()
    rclpy.shutdown()
