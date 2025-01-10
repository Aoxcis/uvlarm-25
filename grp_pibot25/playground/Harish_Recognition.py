import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
colorizer = rs.colorizer()

# FPS set to 30
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.depth
align = rs.align(align_to)

# Segmentation parameters using HSV color space (for green)
color = 60  # Green hue value in HSV
lo = np.array([color - 10, 100, 50])  # Lower bound of green
hi = np.array([color + 10, 255, 255])  # Upper bound of green

# Kernel for morphological operations (to remove small noise)
kernel = np.ones((5, 5), np.uint8)

# Function to get distance to object in the 3D space
def get_distance_to_object(depth_frame, x, y):
    distance = depth_frame.get_distance(x, y)
    return distance

# Function to create a distance heatmap
def create_distance_heatmap(depth_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image_normalized = cv.normalize(depth_image, None, 0, 255, cv.NORM_MINMAX)
    depth_image_color = cv.applyColorMap(depth_image_normalized.astype(np.uint8), cv.COLORMAP_JET)
    return depth_image_color

try:
    while True:
        # RealSense frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not aligned_color_frame:
            continue

        # Get color image from the aligned color frame
        color_image = np.asanyarray(aligned_color_frame.get_data())

        # Convert color image to HSV
        hsv_image = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

        # Create mask to detect green color (tuned for more precise green)
        mask = cv.inRange(hsv_image, lo, hi)

        # Apply morphological operations to clean up the mask
        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=1)

        # Use mask to segment the image
        image_segmented = cv.bitwise_and(color_image, color_image, mask=mask)

        # Find contours of the segmented image
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Prepare distance heatmap
        distance_heatmap = create_distance_heatmap(depth_frame)

        # Iterate over contours and draw bounding boxes for detected green objects
        for contour in contours:
            contour_area = cv.contourArea(contour)

            # Filter out very small contours to avoid noise
            if contour_area < 500:
                continue  # Ignore small contours that are likely noise

            # Get bounding box for the contour
            x, y, w, h = cv.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            distance = get_distance_to_object(depth_frame, center_x, center_y)

            # Draw bounding box around detected green object
            cv.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(color_image, f"Distance: {distance:.2f}m", (x, y + h + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the original color image with detected green objects
        cv.imshow("Green Object Detection", color_image)

        # Display the distance heatmap
        cv.imshow("Distance Heatmap", distance_heatmap)

        # Wait for a key press and exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    pipeline.stop()
    cv.destroyAllWindows()
