import rclpy
from rclpy.node import Node
from std_msgs.msg import String

def oneTalk():
    # Initialize ROS client
    rclpy.init()
    # Create a node
    aNode= Node( "simpleTalker" )
    # Attach a publisher to the node, with a specific type, the name of the topic, a history depth
    aPublisher= aNode.create_publisher( String, 'testTopic', 10 )
    # Create a message to send
    msg = String()
    msg.data = 'Hello World'
    # Add the message to the list of messages to publish
    aPublisher.publish(msg)
    # Activate the ROS client with the node
    # (that will publish the message on testTopic topic)
    rclpy.spin_once(aNode, timeout_sec= 10.0)
    # Clean everything and switch the light off
    aNode.destroy_node()
    rclpy.shutdown()

# Execute the function.
if __name__ == "__main__":
    oneTalk()