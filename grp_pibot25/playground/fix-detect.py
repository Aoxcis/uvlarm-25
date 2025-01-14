#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import sys

class Realsense(Node):
    def __init__(self, fps=30):
        super().__init__('realsense_node')  # Node name
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bridge = CvBridge()

        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

        # Align color frame to depth frame
        self.align_to = rs.stream.depth
        self.align = rs.align(self.align_to)

        # Start the pipeline
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except Exception as e:
            self.get_logger().error(f"Error starting pipeline: {e}")
            exit(1)

        # ROS publishers
        self.color_publisher = self.create_publisher(Image, 'camera/color', 10)
        self.depth_publisher = self.create_publisher(Image, 'camera/depth', 10)
        self.detection_publisher = self.create_publisher(String, 'detection', 10)

        # Initialize background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def read_and_process_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.get_logger().warn("Failed to get frames.")
            return None, None, None

        # Get color and depth images
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame

    def detect_objects(self, color_image, depth_frame):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Mask for detecting green objects
        lower_green = np.array([40, 80, 40])  # Lower bound for green
        upper_green = np.array([75, 255, 255])  # Upper bound for green
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Clean the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.erode(green_mask, kernel, iterations=1)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)

        # Find contours in the green mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) < 150:  # Ignore small objects
                continue

            # Get the bounding box and the aspect ratio of the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # We don't care about the shape anymore, just use a placeholder name
            shape_type = "Object"

            # Calculate the center and distance to the object
            center_x = x + w // 2
            center_y = y + h // 2
            distance = depth_frame.get_distance(center_x, center_y)

            # Store the detected object along with distance
            detected_objects.append((shape_type, x, y, w, h, distance))

        return detected_objects

    def process_and_publish(self):
        color_image, depth_image, depth_frame = self.read_and_process_frames()

        if color_image is None:
            return

        # Detect objects
        detected_objects = self.detect_objects(color_image, depth_frame)

        # Draw bounding boxes and add text
        for idx, (label, x, y, w, h, distance) in enumerate(detected_objects):
            # Use "Object 1", "Object 2", etc.
            object_label = f"Object {idx + 1}"

            # Draw the bounding box and label on the image
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(color_image, f"{object_label} - {distance:.2f}m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Publish detection message
            message = f"{object_label} detected at ({x}, {y}), Distance: {distance:.2f}m"
            self.detection_publisher.publish(String(data=message))

        # Publish the color image
        msg_color = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        msg_color.header.stamp = self.get_clock().now().to_msg()
        msg_color.header.frame_id = "camera_color_frame"
        self.color_publisher.publish(msg_color)

        # Publish the depth image
        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "mono16")
        msg_depth.header.stamp = self.get_clock().now().to_msg()
        msg_depth.header.frame_id = "camera_depth_frame"
        self.depth_publisher.publish(msg_depth)

        # Show the processed image
        cv2.imshow("Object Detection (Shape-based)", color_image)
        
        # Check for 'q' key press to kill the program
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' key to quit
            self.get_logger().info("Kill switch triggered by pressing 'q'. Stopping the program...")
            self.pipeline.stop()
            self.destroy_node()
            rclpy.shutdown()
            cv2.destroyAllWindows()
            sys.exit(0)

    def run(self):
        while rclpy.ok():
            self.process_and_publish()
            rclpy.spin_once(self, timeout_sec=0.001)


def main(args=None):
    rclpy.init(args=args)
    rs_node = Realsense()

    try:
        rs_node.run()
    finally:
        rs_node.pipeline.stop()
        rs_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
