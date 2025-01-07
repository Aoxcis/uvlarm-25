import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Set up alignment to align color with depth
align_to = rs.stream.color
align = rs.align(align_to)

while True:
    # Wait for a coherent set of frames: depth and color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned color frame
    color_frame = aligned_frames.get_color_frame()

    if not color_frame:
        continue

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert the color image to HSV (better for color detection)
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

    # Draw bounding boxes around detected green objects (possible bottles)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small contours
            # Get bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Green Bottle Detection', color_image)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the RealSense pipeline and close any OpenCV windows
pipeline.stop()
cv2.destroyAllWindows()
