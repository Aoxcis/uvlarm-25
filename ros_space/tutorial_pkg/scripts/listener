#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def listen():
    # Initialize ROS node with ROS client
    rclpy.init()
    aNode= Node( "listener" )
    listener= ROSListener(aNode)
    # Start infinite loop
    rclpy.spin(aNode)
    # Clean everything and switch the light off
    aNode.destroy_node()
    rclpy.shutdown()

class ROSListener():

    def __init__(self, rosNode):
        self._logger= rosNode.get_logger()
        self._subscription= rosNode.create_subscription(
            String, 'testTopic',
            self.listener_callback, 10
        )

    def listener_callback(self, msg):
        self._logger.info( 'I heard: ' + msg.data)

if __name__ == '__main__':
    listen()