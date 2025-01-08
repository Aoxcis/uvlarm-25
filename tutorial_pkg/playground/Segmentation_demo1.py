import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
from torchvision import models, transforms

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

        # Load the DeepLabV3 model pre-trained on COCO
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()  # Set model to evaluation mode

        # Define image transformation to match model input size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to 224x224 (DeepLabV3 input size)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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

        return color_image, depth_colormap, depth_image

    def process_green_segmentation(self, color_image):
        """
        This function performs simple color-based segmentation to detect green objects
        using HSV color space.
        """
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV space
        lower_green = np.array([35, 40, 40])  # Lower bound for green
        upper_green = np.array([85, 255, 255])  # Upper bound for green

        # Create a mask to extract the green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Perform morphological operations to clean the mask (optional)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the green regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Variable to store the "most green" contour
        most_green_contour = None
        max_mean_green = 0  # To store the highest mean green intensity

        # Iterate through all contours to find the "most green" object
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ignore small contours
                # Get bounding box around contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the region of the contour from the original image
                roi = color_image[y:y+h, x:x+w]

                # Convert ROI to HSV and calculate the mean green intensity
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_green = np.mean(hsv_roi[:, :, 1])  # Mean value of the Saturation channel

                # If this contour has a higher mean green intensity, update the "most green" object
                if mean_green > max_mean_green:
                    max_mean_green = mean_green
                    most_green_contour = contour

        # If a most green contour is found, draw a bounding box around it
        if most_green_contour is not None:
            x, y, w, h = cv2.boundingRect(most_green_contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return color_image

    def process_with_deep_labv3(self, color_image):
        """
        This function runs the DeepLabV3 model for semantic segmentation.
        It detects all objects, and then we filter for green regions.
        """
        input_image = self.transform(color_image).unsqueeze(0)  # Add batch dimension

        # Run inference with DeepLabV3
        with torch.no_grad():
            output = self.model(input_image)['out'][0]  # Output is a tensor of shape (num_classes, height, width)

        # Convert output tensor to probabilities
        output_predictions = torch.argmax(output, dim=0)  # Get the class with highest score for each pixel

        # Convert output_predictions to a numpy array
        segmentation_mask = output_predictions.cpu().numpy()

        # Create a mask for the green pixels using color-based segmentation
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Resize the green_mask to match the shape of segmentation_mask (224x224)
        green_mask_resized = cv2.resize(green_mask, (segmentation_mask.shape[1], segmentation_mask.shape[0]))

        # Combine the model output with the green detection mask
        green_object_mask = np.logical_and(segmentation_mask == 5, green_mask_resized)  # Assuming class 5 corresponds to green

        # Resize green_object_mask back to the original image size (480x848)
        green_object_mask_resized = cv2.resize(green_object_mask.astype(np.uint8), (color_image.shape[1], color_image.shape[0]))

        # Map to a color image for visualization
        segmented_image = np.zeros_like(color_image)
        segmented_image[green_object_mask_resized == 1] = [0, 255, 0]  # Green color for detected green regions
        segmented_image[green_object_mask_resized == 0] = [0, 0, 0]  # Black for background

        return segmented_image

    def publish_imgs(self):
        color_image, depth_colormap, depth_image = self.read_imgs()

        if color_image is not None:
            # Process the color image to detect green objects using two methods
            color_image = self.process_green_segmentation(color_image)
            color_image_deep = self.process_with_deep_labv3(color_image)

            # Convert color image to ROS message
            msg_color = self.bridge.cv2_to_imgmsg(color_image_deep, "bgr8")
            msg_color.header.stamp = self.get_clock().now().to_msg()
            msg_color.header.frame_id = "camera_color_frame"
            self.color_publisher.publish(msg_color)

            # Show the processed image with bounding box in an OpenCV window
            cv2.imshow("Green Object Segmentation", color_image_deep)

        if depth_colormap is not None:
            # Convert depth image to ROS message
            msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "mono16")
            msg_depth.header.stamp = self.get_clock().now().to_msg()
            msg_depth.header.frame_id = "camera_depth_frame"
            self.depth_publisher.publish(msg_depth)

            # Show the depth map in a separate OpenCV window
            cv2.imshow("Depth Map", depth_colormap)

        # Ensure OpenCV windows are updated
        cv2.waitKey(1)

def process_img(args=None):
    rclpy.init(args=args)
    rs_node = Realsense()

    while rclpy.ok():
        rs_node.publish_imgs()
        rs_node.get_logger().info("Publishing images...")
        rclpy.spin_once(rs_node, timeout_sec=0.001)  # Non-blocking spin to ensure windows update

    # Stop streaming and shutdown node
    rs_node.pipeline.stop()
    rs_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    process_img()
