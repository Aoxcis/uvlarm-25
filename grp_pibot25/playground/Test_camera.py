import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the depth and color streams
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

try:
    # Start the pipeline
    print("Starting pipeline...")
    pipeline.start(config)
    print("Pipeline started successfully.")
    
    while True:
        # Wait for frames with a timeout of 10 seconds (10000 ms)
        print("Waiting for frames...")
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        
        if frames:
            print("Frames received.")
            
            # Get the depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Error: Missing depth or color frame.")
                continue

            print("Depth and color frames received.")
            print(f"Depth Frame: {depth_frame.get_width()}x{depth_frame.get_height()}")
            print(f"Color Frame: {color_frame.get_width()}x{color_frame.get_height()}")

            # Here you could add further processing of the frames (e.g., visualization, distance measurement)
            
        else:
            print("Error: No frames received within the timeout period.")
        
except Exception as e:
    # Catch any errors and display them
    print(f"An error occurred: {e}")
    
finally:
    # Stop the pipeline when done
    pipeline.stop()
    print("Pipeline stopped.")
