---
sidebar_position: 1
title: "Introduction to ROS 2"
---

# Introduction to ROS 2: The Foundation of Robotic Systems

This tutorial provides a comprehensive introduction to ROS 2 (Robot Operating System 2), the middleware that serves as the foundation for modern robotics development. You'll learn about the core concepts, architecture, and practical implementation of ROS 2 systems.

## What is ROS 2?

ROS 2 is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. Unlike traditional monolithic frameworks, ROS 2 is designed to be distributed, allowing different parts of your robot to run on different computers or processors.

### Key Advantages of ROS 2

1. **Distributed Architecture**: Components can run on different machines and communicate seamlessly
2. **Real-time Support**: Better support for real-time applications compared to ROS 1
3. **Security**: Built-in security features for safe robot operation
4. **Quality of Service (QoS)**: Configurable communication patterns for different requirements
5. **Cross-platform**: Runs on Linux, macOS, Windows, and embedded systems
6. **Language Support**: Native support for C++, Python, and other languages

## ROS 2 Architecture Overview

### Nodes
Nodes are the fundamental building blocks of ROS 2 applications. Each node performs a specific task and communicates with other nodes through messages. Nodes can be written in different programming languages and run on different machines.

```python
# Example ROS 2 node structure
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Messages
Topics are named buses over which nodes exchange messages. The publisher-subscriber pattern allows for asynchronous communication between nodes.

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Services
Services provide a request-response communication pattern. A node sends a request and waits for a response from a service server.

```python
# Service client example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Setting Up Your ROS 2 Environment

### Installation
The recommended installation method for ROS 2 is through packages. For Ubuntu, you can install ROS 2 Humble Hawksbill (LTS) with:

```bash
# Setup locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup
After installation, source the ROS 2 environment:

```bash
source /opt/ros/humble/setup.bash
```

For convenience, add this to your `.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating Your First ROS 2 Package

### Using colcon
ROS 2 uses `colcon` as the build system. Create a workspace and your first package:

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Create a new package
colcon create pkg --template-args 'ament_python' my_robot_pkg --dependencies rclpy std_msgs geometry_msgs sensor_msgs
```

### Package Structure
A typical ROS 2 package contains:

```
my_robot_pkg/
├── package.xml          # Package manifest
├── CMakeLists.txt       # Build configuration (for C++)
├── setup.py             # Python setup
├── setup.cfg            # Installation configuration
├── my_robot_pkg/        # Python module
│   ├── __init__.py
│   └── my_node.py
└── test/                # Test files
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

## Core ROS 2 Tools

### ros2 run
Execute a node within a package:

```bash
ros2 run my_robot_pkg my_node
```

### ros2 topic
Inspect and interact with topics:

```bash
# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo /topic_name std_msgs/msg/String

# Publish to a topic
ros2 topic pub /topic_name std_msgs/msg/String "data: 'Hello'"
```

### ros2 service
Interact with services:

```bash
# List all services
ros2 service list

# Call a service
ros2 service call /service_name example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

### ros2 node
Manage nodes:

```bash
# List all active nodes
ros2 node list

# Get information about a node
ros2 node info /node_name
```

## Quality of Service (QoS) in ROS 2

QoS profiles allow you to configure communication behavior based on your application's requirements:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)

# Use the QoS profile when creating a publisher
publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Common ROS 2 Patterns

### Launch Files
Launch files allow you to start multiple nodes with a single command:

```python
# my_robot_pkg/launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='my_node',
            name='my_node',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ]
        ),
        Node(
            package='another_pkg',
            executable='another_node',
            name='another_node'
        )
    ])
```

### Parameters
Use parameters to configure your nodes without recompiling:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Robot name: {self.robot_name}, Max velocity: {self.max_velocity}')
```

## Best Practices for ROS 2 Development

1. **Modular Design**: Break your robot system into logical, independent nodes
2. **Clear Naming**: Use descriptive names for topics, services, and nodes
3. **Error Handling**: Implement proper error handling and logging
4. **Resource Management**: Properly clean up resources when nodes are destroyed
5. **Documentation**: Document your nodes, topics, and services clearly
6. **Testing**: Write tests for your nodes and system integration

## Acceptance Scenarios

The introduction to ROS 2 is understood when:

**Scenario 1**: As a robotics developer, when I create a new ROS 2 node, then it should properly initialize, perform its function, and handle cleanup when terminated.

**Scenario 2**: As a robotics engineer, when I set up communication between nodes, then they should successfully exchange messages through topics, services, or actions with appropriate QoS settings.

**Scenario 3**: As a system architect, when I design a robotic system using ROS 2, then it should follow best practices for modularity, naming, and error handling.

## Next Steps

Once you have completed this introduction to ROS 2, continue with more advanced topics in Module 1, including detailed tutorials on publisher-subscriber patterns, services, and parameter management. These concepts form the foundation for building complex robotic systems with distributed intelligence.