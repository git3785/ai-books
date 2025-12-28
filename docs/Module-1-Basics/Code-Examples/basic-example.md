---
sidebar_position: 3
title: "Basic ROS 2 Example"
---

# Basic ROS 2 Example: Simple Publisher and Subscriber

This code example demonstrates the fundamental ROS 2 communication pattern using a publisher-subscriber model. This is the "Hello World" equivalent for ROS 2 and forms the basis for more complex robotic systems.

## Publisher Node Implementation

The publisher node continuously sends messages to a topic. This could represent sensor data, robot state, or any other streaming information.

```python
#!/usr/bin/env python3

"""
Simple publisher node that sends messages to a topic
This represents the basic building block of ROS 2 communication
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        
        # Create a publisher for the 'topic' topic with String messages
        # The second parameter (10) is the queue size for messages
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        
        # Set up a timer to call the timer_callback method every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Counter to keep track of published messages
        self.i = 0

    def timer_callback(self):
        """This method is called by the timer at regular intervals"""
        # Create a String message
        msg = String()
        msg.data = f'Hello World: {self.i}'
        
        # Publish the message
        self.publisher_.publish(msg)
        
        # Log the published message to the console
        self.get_logger().info(f'Publishing: "{msg.data}"')
        
        # Increment the counter
        self.i += 1


def main(args=None):
    """Main function to initialize and run the publisher node"""
    # Initialize the ROS 2 communication
    rclpy.init(args=args)
    
    # Create the publisher node
    minimal_publisher = MinimalPublisher()
    
    try:
        # Keep the node running until interrupted
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up resources
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Subscriber Node Implementation

The subscriber node listens to messages from the publisher. This represents how different parts of a robot system receive information from other components.

```python
#!/usr/bin/env python3

"""
Simple subscriber node that receives messages from a topic
This demonstrates the other side of the publisher-subscriber pattern
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        
        # Create a subscription to the 'topic' topic
        # The callback method will be called when a message is received
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)  # Queue size
        
        # Make sure the subscription is properly created
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        """This method is called when a message is received"""
        # Log the received message to the console
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """Main function to initialize and run the subscriber node"""
    # Initialize the ROS 2 communication
    rclpy.init(args=args)
    
    # Create the subscriber node
    minimal_subscriber = MinimalSubscriber()
    
    try:
        # Keep the node running to receive messages
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up resources
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Complete Package Structure

Here's how to structure these files in a complete ROS 2 package:

### Package.xml
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>basic_ros2_examples</name>
  <version>0.0.0</version>
  <description>Basic ROS 2 examples for learning purposes</description>
  <maintainer email="example@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup.py
```python
from setuptools import setup

package_name = 'basic_ros2_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Basic ROS 2 examples for learning purposes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = basic_ros2_examples.publisher_member_function:main',
            'listener = basic_ros2_examples.subscriber_member_function:main',
        ],
    },
)
```

### Setup.cfg
```
[develop]
script-dir=$base/lib/basic_ros2_examples
[install]
install-scripts=$base/lib/basic_ros2_examples
```

## Running the Example

### Terminal 1 (Publisher):
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Navigate to your workspace
cd ~/ros2_ws

# Source the workspace
source install/setup.bash

# Run the publisher node
ros2 run basic_ros2_examples talker
```

### Terminal 2 (Subscriber):
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Navigate to your workspace
cd ~/ros2_ws

# Source the workspace
source install/setup.bash

# Run the subscriber node
ros2 run basic_ros2_examples listener
```

## Advanced Publisher with Custom Message

Here's a more complex example that demonstrates creating and using custom messages:

```python
#!/usr/bin/env python3

"""
Advanced publisher with custom message structure
Useful for humanoid robotics applications
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Header
import time


class AdvancedPublisher(Node):

    def __init__(self):
        super().__init__('advanced_publisher')
        
        # Publisher for joint states
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_states', 10)
        
        # Timer for publishing at regular intervals
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_joint_states)
        
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example: 6 joints
        self.time_step = 0

    def publish_joint_states(self):
        """Publish simulated joint positions"""
        msg = Float64MultiArray()
        
        # Update joint positions (simulating movement)
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = 0.5 * 3.14159 * (i + 1) * (self.time_step % 100) / 100.0
            
        msg.data = self.joint_positions
        
        self.joint_pub.publish(msg)
        self.get_logger().info(f'Published joint states: {msg.data}')
        
        self.time_step += 1


def main(args=None):
    rclpy.init(args=args)
    advanced_publisher = AdvancedPublisher()
    
    try:
        rclpy.spin(advanced_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        advanced_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Subscriber with Multiple Topics

```python
#!/usr/bin/env python3

"""
Advanced subscriber that listens to multiple topics
Demonstrates how a robot controller might integrate multiple data sources
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
import time


class AdvancedSubscriber(Node):

    def __init__(self):
        super().__init__('advanced_subscriber')
        
        # Subscriptions to different topics
        self.joint_sub = self.create_subscription(
            Float64MultiArray,
            'joint_states',
            self.joint_callback,
            10)
        
        self.status_sub = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10)
        
        # Store latest values
        self.latest_joint_states = None
        self.latest_status = "unknown"
        
        # Timer to process integrated data
        self.timer = self.create_timer(0.5, self.process_integrated_data)

    def joint_callback(self, msg):
        """Handle joint state updates"""
        self.latest_joint_states = msg.data
        self.get_logger().info(f'Received joint states: {msg.data}')

    def status_callback(self, msg):
        """Handle status updates"""
        self.latest_status = msg.data
        self.get_logger().info(f'Received status: {msg.data}')

    def process_integrated_data(self):
        """Process integrated data from multiple sources"""
        if self.latest_joint_states is not None:
            avg_position = sum(self.latest_joint_states) / len(self.latest_joint_states) if self.latest_joint_states else 0
            self.get_logger().info(
                f'Integrated data - Status: {self.latest_status}, '
                f'Avg Joint Position: {avg_position:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    advanced_subscriber = AdvancedSubscriber()
    
    try:
        rclpy.spin(advanced_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        advanced_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Configuration

For robotics applications, you may need to configure QoS settings:

```python
#!/usr/bin/env python3

"""
Publisher with custom QoS settings
Important for real-time robotics applications
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class QoSPublisher(Node):

    def __init__(self):
        super().__init__('qos_publisher')
        
        # Create a QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        
        # Create publisher with custom QoS
        self.publisher_ = self.create_publisher(String, 'qos_topic', qos_profile)
        
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'QoS Message: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published with QoS: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    qos_publisher = QoSPublisher()
    
    try:
        rclpy.spin(qos_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        qos_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File for Running Both Nodes

Create a launch file to start both publisher and subscriber simultaneously:

```python
# basic_ros2_examples/launch/basic_example_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='basic_ros2_examples',
            executable='talker',
            name='publisher_node',
            output='screen'
        ),
        Node(
            package='basic_ros2_examples',
            executable='listener',
            name='subscriber_node',
            output='screen'
        )
    ])
```

## Running with Launch File

```bash
# Source your workspace
source install/setup.bash

# Run both nodes with launch file
ros2 launch basic_ros2_examples basic_example_launch.py
```

## Key Concepts Demonstrated

1. **Node Creation**: How to create ROS 2 nodes with proper lifecycle management
2. **Publishing**: How to publish messages to topics
3. **Subscribing**: How to subscribe to topics and handle incoming messages
4. **Timers**: How to execute code at regular intervals
5. **Logging**: How to log information for debugging
6. **Resource Management**: Proper cleanup of resources
7. **QoS Configuration**: How to configure Quality of Service settings
8. **Launch Files**: How to run multiple nodes simultaneously

## Application to Physical AI & Humanoid Robotics

This basic example forms the foundation for more complex humanoid robotics systems:

- Joint state publishers can publish actual sensor readings from robot joints
- Subscriber nodes can receive commands from high-level planners
- Multiple sensors can publish data to different topics simultaneously
- Control nodes can subscribe to multiple topics to integrate sensor data
- QoS settings ensure critical control messages are delivered reliably

## Acceptance Scenarios

The basic ROS 2 example is properly implemented when:

**Scenario 1**: As a ROS 2 developer, when I run the publisher and subscriber nodes, then messages should flow correctly from publisher to subscriber with appropriate logging.

**Scenario 2**: As a robotics engineer, when I implement the publisher-subscriber pattern, then it should follow ROS 2 best practices for node lifecycle and resource management.

**Scenario 3**: As a system integrator, when I combine multiple nodes with different QoS requirements, then they should communicate appropriately based on their configuration.

## Next Steps

Once you have mastered this basic ROS 2 example, explore more complex patterns in Module 1, including services, actions, and parameter management. These concepts form the foundation for building sophisticated robotic systems with distributed intelligence.