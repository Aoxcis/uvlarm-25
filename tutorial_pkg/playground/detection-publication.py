import cv2
import numpy as np
import pyrealsense2 as rs
import os

image_path = 'bottle_template2.jpg'

# Check if the file exists at the given path
if not os.path.isfile(image_path):
    print(f"Error: The image file does not exist at the path: {image_path}")
else:
    print(f"Image file exists at the path: {image_path}")

# Load the bottle template (this image should be a clear image of the bottle)
template = cv2.imread(image_path, 0)  # Load the template in grayscale

# Resize the template to a desired size (e.g., 50% of its original size)
scale_factor = 0.5
new_width = int(template.shape[1] * scale_factor)
new_height = int(template.shape[0] * scale_factor)
template_resized = cv2.resize(template, (new_width, new_height))

# Ensure the template size is smaller than the image size
# Get the frame size
frame_width = 640  # Or dynamically get this from the RealSense camera
frame_height = 480  # Or dynamically get this from the RealSense camera

# Check if the resized template is larger than the image frame, and resize again if necessary
if template_resized.shape[1] > frame_width or template_resized.shape[0] > frame_height:
    print("Template is larger than the frame. Resizing the template.")
    scale_factor = min(frame_width / template.shape[1], frame_height / template.shape[0])
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template_resized = cv2.resize(template, (new_width, new_height))

# Get the width and height of the resized template
w, h = template_resized.shape[::-1]

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream RGB and Depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB video stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start streaming
pipeline.start(config)

# Set up RealSense alignment (RGB <-> Depth)
align_to = rs.stream.color
align = rs.align(align_to)

while True:
    # Wait for a coherent set of frames: depth and color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame:
        continue

    # Convert color image to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert the frame to grayscale (template matching requires grayscale images)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching to find the bottle in the frame
    res = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)

    # Define a threshold to consider it a match
    threshold = 0.15  # Adjusted for better accuracy
    loc = np.where(res >= threshold)  # Locations where the template matches

    # Check if there are any matches, and only then draw the rectangles
    if loc[0].size > 0:  # Only proceed if matches are found
        for pt in zip(*loc[::-1]):
            # Draw rectangles around the matches
            cv2.rectangle(color_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    else:
        print("No matches found.")

    # Display the resulting frame with the detected bottle
    cv2.imshow('Bottle Detection', color_image)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the RealSense pipeline and close any OpenCV windows
pipeline.stop()
cv2.destroyAllWindows()
