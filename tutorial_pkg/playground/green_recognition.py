import cv2
import numpy as np

# Start capturing video from the laptop camera
cap = cv2.VideoCapture(0)  # Use '0' for default laptop camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Set camera resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([35, 40, 40])  # Lower bound for green (H: 35-85)
    upper_green = np.array([85, 255, 255])  # Upper bound for green (H: 35-85)

    # Create a binary mask where green colors are detected
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform some morphological operations to clean the mask (optional)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected green regions (possible bottles)
    for contour in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(contour) > 500:
            # Get the bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the rectangle around the green object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with detected green bottle
    cv2.imshow('Green Bottle Detection', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
