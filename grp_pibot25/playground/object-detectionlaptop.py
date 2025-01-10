import cv2
import numpy as np
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
# Get the frame size (set to 1280x720 as per your laptop's camera resolution)
frame_width = 1280  
frame_height = 720  

# Check if the resized template is larger than the image frame, and resize again if necessary
if template_resized.shape[1] > frame_width or template_resized.shape[0] > frame_height:
    print("Template is larger than the frame. Resizing the template.")
    scale_factor = min(frame_width / template.shape[1], frame_height / template.shape[0])
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template_resized = cv2.resize(template, (new_width, new_height))

# Get the width and height of the resized template
w, h = template_resized.shape[::-1]

# Start capturing video from the laptop's camera
cap = cv2.VideoCapture(0)  # Use '0' for default laptop camera, or '1'/'2' if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Set camera resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
    # Read a frame from the camera
    ret, color_image = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to grayscale (template matching requires grayscale images)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching to find the bottle in the frame
    res = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)

    # Define a threshold to consider it a match
    threshold = 0.9  # Adjusted for better accuracy
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

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
