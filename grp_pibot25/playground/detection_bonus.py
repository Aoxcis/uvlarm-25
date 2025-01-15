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

        # Memory to store previously detected objects by ID
        self.detected_memory = {}
        self.next_object_id = 1  # Unique ID for each detected object

        # Load template images for template matching (you need to provide the actual templates)
        self.templates = {
            "Phantom Ghost": cv2.imread("phantom_ghost_template.png", cv2.IMREAD_GRAYSCALE),
            "Nuka-Cola Bottle": cv2.imread("nuka_cola_bottle_template.png", cv2.IMREAD_GRAYSCALE)
        }

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

    def detect_objects_contour(self, color_image, depth_frame):
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

            # Apply shape filters
            if len(approx) > 4:  # More than 4 vertices indicates a rounded shape (possibly bottle)
                shape_type = "Phantom Ghost"
            else:  # Fewer vertices indicates a more jagged shape (possibly ghost)
                shape_type = "Nuka-Cola Bottle"

            # Calculate the center and distance to the object
            center_x = x + w // 2
            center_y = y + h // 2
            distance = depth_frame.get_distance(center_x, center_y)

            detected_objects.append((shape_type, x, y, w, h, distance, center_x, center_y))

        return detected_objects

    def detect_objects_template(self, color_image, depth_frame):
        detected_objects = []

        # Convert the color image to grayscale for template matching
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Iterate over all the templates and try to match them
        for label, template in self.templates.items():
            res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8  # You can adjust this threshold
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):  # Get coordinates of matched regions
                x, y = pt
                w, h = template.shape[::-1]

                # Calculate the center and distance to the object
                center_x = x + w // 2
                center_y = y + h // 2
                distance = depth_frame.get_distance(center_x, center_y)

                detected_objects.append((label, x, y, w, h, distance, center_x, center_y))

        return detected_objects

    def store_or_recognize_object(self, detected_objects):
        for label, x, y, w, h, distance, center_x, center_y in detected_objects:
            recognized = False

            # Try to match the detected object with previously stored ones
            for obj_id, obj_data in self.detected_memory.items():
                prev_label, prev_center_x, prev_center_y = obj_data

                # If the object is close enough to a previous detection, consider it the same object
                if abs(center_x - prev_center_x) < 50 and abs(center_y - prev_center_y) < 50:
                    recognized = True
                    self.get_logger().info(f"Recognized {label} as {prev_label} with ID {obj_id}.")
                    break

            # If not recognized, store it as a new object
            if not recognized:
                obj_id = self.next_object_id
                self.detected_memory[obj_id] = (label, center_x, center_y)
                self.get_logger().info(f"New {label} detected and stored with ID {obj_id}.")
                self.next_object_id += 1

            # Publish detection message with object ID
            message = f"{label} (ID {obj_id}) detected at ({x}, {y}), Distance: {distance:.2f}m"
            self.detection_publisher.publish(String(data=message))

    def process_and_publish(self):
        color_image, depth_image, depth_frame = self.read_and_process_frames()

        if color_image is None:
            return

        # Detect objects using contours first
        detected_objects = self.detect_objects_contour(color_image, depth_frame)

        # If no objects detected, fall back to template matching
        if not detected_objects:
            detected_objects = self.detect_objects_template(color_image, depth_frame)

        # Store or recognize objects from memory
        self.store_or_recognize_object(detected_objects)

        # Draw bounding boxes and add text with object ID
        for label, x, y, w, h, distance, _, _ in detected_objects:
            # Draw the bounding box
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add text (e.g., label with ID and distance) above the bounding box
            message = f"{label} - ID {self.next_object_id - 1} - {distance:.2f}m"
            cv2.putText(color_image, message, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
        cv2.imshow("Object Detection (Combined Method)", color_image)

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
