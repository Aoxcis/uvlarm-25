import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class GreenNukaColaDetection(Node):
    def __init__(self, fps=30):
        super().__init__('green_nuka_cola_detection_node')  # Node name
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.bridge = CvBridge()

        # Configure the RealSense streams (color and depth)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)

        # Start the pipeline
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except Exception as e:
            self.get_logger().error(f"Error starting the pipeline: {e}")
            exit(1)

        # Create image publishers (optional)
        self.color_publisher = self.create_publisher(Image, 'camera/color', 10)
        self.depth_publisher = self.create_publisher(Image, 'camera/depth', 10)

    def read_imgs(self):
        # Capture frames from the camera
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("Failed to get frames from camera.")
            return None, None

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def detect_green_bottle(self, color_image):
        # Convert the color image to HSV (better for color detection)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV space
        lower_green = np.array([35, 40, 40])  # Lower bound for green
        upper_green = np.array([85, 255, 255])  # Upper bound for green

        # Create a mask to extract the green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Optional: Perform morphological operations to clean the mask (removing small noise)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the green regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Variable to store the "largest" green region (which could be the bottle)
        largest_contour = None
        max_area = 0

        # Iterate through all contours to find the largest green contour
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small contours (can be adjusted)
                # Get bounding box around contour
                x, y, w, h = cv2.boundingRect(contour)

                # If this contour is the largest one, store it
                if cv2.contourArea(contour) > max_area:
                    max_area = cv2.contourArea(contour)
                    largest_contour = contour

        # If a large enough green contour is found, draw a bounding box around it
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return color_image, mask

    def publish_imgs(self, color_image, depth_image):
        # Convert the color image to ROS message
        msg_color = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        msg_color.header.stamp = self.get_clock().now().to_msg()
        msg_color.header.frame_id = "camera_color_frame"
        self.color_publisher.publish(msg_color)

        # Convert the depth image to ROS message
        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "mono16")
        msg_depth.header.stamp = self.get_clock().now().to_msg()
        msg_depth.header.frame_id = "camera_depth_frame"
        self.depth_publisher.publish(msg_depth)

    def detect_and_publish(self):
        # Main loop for detection and publishing
        while rclpy.ok():
            color_image, depth_image = self.read_imgs()

            if color_image is not None:
                # Perform green bottle detection
                color_image, mask = self.detect_green_bottle(color_image)

                # Publish the processed images
                self.publish_imgs(color_image, depth_image)

                # Display the result with bounding boxes
                cv2.imshow("Green Nuka-Cola Bottle Detection", color_image)

                # Optional: Display the mask showing detected green regions
                # cv2.imshow("Green Mask", mask)

            # Wait for keypress (exit loop if 'q' is pressed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop the pipeline and close OpenCV windows
        self.pipeline.stop()
        cv2.destroyAllWindows()


def process_detection(args=None):
    rclpy.init(args=args)
    rs_node = GreenNukaColaDetection()

    # Start the detection and publishing loop
    rs_node.get_logger().info("Starting green Nuka-Cola bottle detection...")
    rs_node.detect_and_publish()

    # Shutdown ROS
    rclpy.shutdown()


if __name__ == '__main__':
    process_detection()
