#!/usr/bin/python3
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from geometry_msgs.msg import Pose
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class MapExplorer:
    def __init__(self):
        self.node = rclpy.create_node('map_explorer')
        self.map_sub = self.node.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.costmap_sub = self.node.create_subscription(
            OccupancyGrid, '/move_base/global_costmap/costmap', self.costmap_callback, 10)
        self._PubGoal = self.node.create_publisher(Pose, '/goal', 10)

        self.map_data = None
        self.costmap_data = None

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)

    def costmap_callback(self, msg):
        self.costmap_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)

    def get_unexplored_frontiers(self):
        frontiers = []
        for i in range(1, self.map_data.shape[0] - 1):
            for j in range(1, self.map_data.shape[1] - 1):
                if self.map_data[i, j] == 255:  # Unknown region
                    # Check neighboring cells for free (0) or occupied (100)
                    neighbors = [
                        self.map_data[i-1, j], self.map_data[i+1, j],
                        self.map_data[i, j-1], self.map_data[i, j+1]
                    ]   
                    if any(n in [0, 100] for n in neighbors):
                        frontiers.append((i, j))
        return frontiers
    
    def publish_goal(self):
        # Get the current time for transformations
        current_time = rclpy.time.Time()

        try:
            # Lookup the transform from 'odom' to 'base_link'
            stamped_transform = self.tf_buffer.lookup_transform(
                self.local_frame, 'odom', current_time)

            # Get an unexplored frontier
            frontiers = self.get_unexplored_frontiers()
            if frontiers:
                # Pick the first frontier (you can implement a more complex strategy here)
                goal_x, goal_y = frontiers[0]
                goal_pose = Pose()
                goal_pose.position.x = goal_x  # Convert map cell to actual position
                goal_pose.position.y = goal_y

                # Transform the goal from the global frame (map) to the local frame
                local_goal = do_transform_pose(goal_pose, stamped_transform)

                # Publish or use the transformed local goal
                self.node.get_logger().info(f"Publishing goal: {local_goal}")
                self._PubGoal.publish(local_goal.pose)
                
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.node.get_logger().info(f"Transform failed: {e}")

