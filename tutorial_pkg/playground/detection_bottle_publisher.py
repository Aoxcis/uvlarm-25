import cv2
import numpy as np
import pyrealsense2 as rs
import os
from std_msgs.msg import String
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

image_path = 'bottle_template.png'

class Realsense(Node):
    def __init__(self, fps=60):
        super().__init__('detection_node')  # Node name
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

        # Create detection publisher
        self.detection_publisher = self.create_publisher(String, 'detection', 10)

    def detect_bottle(self, template_resized, w, h):
        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            # Wait for a coherent set of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                continue

            # Convert color image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Convert the frame to grayscale (template matching requires grayscale images)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Apply template matching to find the bottle in the frame
            res = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)

            # Define a threshold to consider it a match
            threshold = 0.9
            loc = np.where(res >= threshold)

            # Check if there are any matches, and only then draw the rectangles
            if loc[0].size > 0:
                for pt in zip(*loc[::-1]):
                    # Draw rectangles around the matches
                    cv2.rectangle(color_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                    self.detection_publisher.publish(String(data=f"Bottle detected at: {pt}"))
            else:
                self.detection_publisher.publish(String(data="No matches found."))

            # Display the resulting frame with the detected bottle
            cv2.imshow('Bottle Detection', color_image)

            # Break the loop when the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the RealSense pipeline and close any OpenCV windows
        self.pipeline.stop()
        cv2.destroyAllWindows()


def lancer_detection(args=None):
    rclpy.init(args=args)
    rs_node = Realsense()

    if not os.path.isfile(image_path):
        rs_node.get_logger().error(f"The image file does not exist at the path: {image_path}")
        rclpy.shutdown()
        return

    rs_node.get_logger().info(f"Image file exists at the path: {image_path}")

    # Load the bottle template
    template = cv2.imread(image_path, 0)

    # Resize the template
    scale_factor = 0.5
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template_resized = cv2.resize(template, (new_width, new_height))

    # Get the width and height of the resized template
    w, h = template_resized.shape[::-1]

    try:
        rs_node.detect_bottle(template_resized, w, h)
    except Exception as e:
        rs_node.get_logger().error(f"Error during detection: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    lancer_detection()
