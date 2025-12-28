---
sidebar_position: 3
title: "Intermediate Simulation Examples"
---

# Intermediate Simulation Examples: Gazebo and Unity Implementation

This code example demonstrates intermediate-level digital twin implementations using both Gazebo and Unity simulation environments for humanoid robotics. These examples build upon basic ROS 2 concepts to show how to create sophisticated virtual environments that accurately represent physical robots.

## Gazebo Simulation Example

### Robot Model Configuration (URDF)

First, let's create a detailed humanoid robot model configuration for Gazebo:

```xml
<?xml version="1.0" ?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include gazebo_ros_control plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="base_torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.5"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>

  <!-- Sensors -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.02 0.04 0.02"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.04 0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo sensor plugin -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera">
      <update_rate>30.0</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <gazebo reference="torso">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <frame_name>torso</frame_name>
        <topic_name>imu/data</topic_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
</robot>
```

### Gazebo World Configuration

Create a world file for your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="humanoid_world">
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Robot -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Objects for interaction -->
    <model name="table">
      <pose>-1 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="ball">
      <pose>-0.5 0 1.0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.005</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.005</iyy>
            <iyz>0.0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Unity Simulation Example

### Unity Robot Controller Script

Create a Unity script to control the humanoid robot in simulation:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Control_msgs;

public class UnityHumanoidController : MonoBehaviour
{
    ROSConnection ros;
    
    // Robot joint transforms
    [Header("Robot Joints")]
    public Transform head;
    public Transform torso;
    public Transform leftUpperArm;
    public Transform rightUpperArm;
    public Transform leftLowerArm;
    public Transform rightLowerArm;
    
    // Joint position storage
    Dictionary<string, float> jointPositions = new Dictionary<string, float>();
    
    // Topics
    string jointStatesTopic = "joint_states";
    string cameraImageTopic = "unity_camera/image_raw";
    
    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        
        // Register to receive joint commands
        ros.Subscribe<sensor_msgs.JointState>(jointStatesTopic, OnJointStateReceived);
        
        // Start camera publisher coroutine
        StartCoroutine(PublishCameraImage());
    }
    
    // Update is called once per frame
    void Update()
    {
        // Update robot joints based on received commands
        UpdateRobotJoints();
    }
    
    void OnJointStateReceived(sensor_msgs.JointState jointState)
    {
        // Update joint positions dictionary
        for (int i = 0; i < jointState.name.Length; i++)
        {
            if (i < jointState.position.Length)
            {
                jointPositions[jointState.name[i]] = (float)jointState.position[i];
            }
        }
    }
    
    void UpdateRobotJoints()
    {
        // Update head rotation
        if (jointPositions.ContainsKey("neck_joint"))
        {
            head.localRotation = Quaternion.Euler(0, jointPositions["neck_joint"] * Mathf.Rad2Deg, 0);
        }
        
        // Update left arm
        if (jointPositions.ContainsKey("left_shoulder_joint"))
        {
            leftUpperArm.localRotation = Quaternion.Euler(0, jointPositions["left_shoulder_joint"] * Mathf.Rad2Deg, 0);
        }
        
        // Update right arm
        if (jointPositions.ContainsKey("right_shoulder_joint"))
        {
            rightUpperArm.localRotation = Quaternion.Euler(0, jointPositions["right_shoulder_joint"] * Mathf.Rad2Deg, 0);
        }
    }
    
    IEnumerator PublishCameraImage()
    {
        yield return new WaitForSeconds(1.0f); // Wait for initialization
        
        Camera unityCamera = GetComponent<Camera>();
        if (unityCamera == null)
        {
            Debug.LogError("Camera component not found on this GameObject!");
            yield break;
        }
        
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = unityCamera.targetTexture;
        
        int width = unityCamera.targetTexture ? unityCamera.targetTexture.width : 640;
        int height = unityCamera.targetTexture ? unityCamera.targetTexture.height : 480;
        
        while (true)
        {
            // Capture image from Unity camera
            Texture2D imageTex = new Texture2D(width, height, TextureFormat.RGB24, false);
            imageTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            imageTex.Apply();
            
            // Convert to ROS Image message
            sensor_msgs.Image imageMsg = new sensor_msgs.Image();
            imageMsg.header = new std_msgs.Header();
            imageMsg.header.stamp = new builtin_interfaces.Time();
            imageMsg.header.frame_id = "unity_camera_optical_frame";
            
            imageMsg.height = (uint)height;
            imageMsg.width = (uint)width;
            imageMsg.encoding = "rgb8";
            imageMsg.is_bigendian = 0;
            imageMsg.step = (uint)(width * 3); // 3 bytes per pixel for RGB
            
            // Convert texture to byte array
            byte[] imageData = imageTex.EncodeToPNG();
            imageMsg.data = imageData;
            
            // Publish the image
            ros.Publish(cameraImageTopic, imageMsg);
            
            // Clean up
            Destroy(imageTex);
            
            // Wait for next frame (30 FPS)
            yield return new WaitForSeconds(1.0f / 30.0f);
        }
        
        RenderTexture.active = currentRT;
    }
}
```

### Unity Environment Setup Script

Create a script to initialize the Unity simulation environment:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class UnitySimulationSetup : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;
    
    [Header("Simulation Settings")]
    public float simulationSpeed = 1.0f;
    public bool usePhysics = true;
    
    void Start()
    {
        // Initialize ROS connection
        ROSConnection ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        
        Debug.Log($"Unity simulation initialized with ROS at {rosIP}:{rosPort}");
        
        // Set physics time scale
        Time.timeScale = simulationSpeed;
        
        // Configure physics settings
        if (!usePhysics)
        {
            Physics.autoSimulation = false;
        }
    }
    
    void Update()
    {
        // Additional simulation updates can go here
    }
    
    public void ResetSimulation()
    {
        // Reset all objects to initial positions
        foreach (Transform child in transform)
        {
            if (child.CompareTag("Resettable"))
            {
                // Reset position and rotation to initial values
                // Implementation depends on your specific needs
            }
        }
    }
}
```

## ROS 2 Control Node Example

Create a ROS 2 node to control the simulated robot:

```python
#!/usr/bin/env python3

"""
ROS 2 control node for simulated humanoid robot
This node sends commands to both Gazebo and Unity simulations
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import time

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Publisher for status
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        
        # Timer for publishing joint commands
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_joint_commands)
        
        # Joint position storage
        self.joint_positions = {
            'neck_joint': 0.0,
            'left_shoulder_joint': 0.0,
            'right_shoulder_joint': 0.0,
            'left_elbow_joint': 0.0,
            'right_elbow_joint': 0.0,
        }
        
        self.time_step = 0
        self.get_logger().info('Humanoid Controller node initialized')

    def publish_joint_commands(self):
        """Publish joint commands for the simulated robot"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        # Update joint positions with time-varying patterns
        self.time_step += 0.1
        
        # Create oscillating patterns for different joints
        self.joint_positions['neck_joint'] = 0.3 * math.sin(self.time_step * 0.5)
        self.joint_positions['left_shoulder_joint'] = 0.4 * math.sin(self.time_step * 0.7)
        self.joint_positions['right_shoulder_joint'] = 0.4 * math.sin(self.time_step * 0.7 + math.pi)
        self.joint_positions['left_elbow_joint'] = 0.2 * math.sin(self.time_step * 0.9)
        self.joint_positions['right_elbow_joint'] = 0.2 * math.sin(self.time_step * 0.9 + math.pi)
        
        # Set message fields
        msg.name = list(self.joint_positions.keys())
        msg.position = [float(pos) for pos in self.joint_positions.values()]
        
        # Publish the joint state
        self.joint_pub.publish(msg)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"Joint positions updated - Step: {self.time_step:.2f}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f'Published joint commands: {dict(zip(msg.name, msg.position))}')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Humanoid Controller node')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Launch File

Create a launch file to start the complete simulation:

```python
# humanoid_simulation/launch/simulation_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Use Gazebo simulation'
    )
    
    use_unity_arg = DeclareLaunchArgument(
        'use_unity',
        default_value='false',
        description='Use Unity simulation'
    )
    
    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('humanoid_simulation'),
                'worlds',
                'humanoid_world.world'
            ])
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_gazebo'))
    )
    
    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro ', 
                PathJoinSubstitution([
                    FindPackageShare('humanoid_simulation'),
                    'urdf',
                    'humanoid_robot.urdf.xacro'
                ])
            ])
        }]
    )
    
    # Joint state publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )
    
    # Robot controller node
    robot_controller = Node(
        package='humanoid_simulation',
        executable='humanoid_controller',
        name='humanoid_controller',
        output='screen'
    )
    
    return LaunchDescription([
        use_gazebo_arg,
        use_unity_arg,
        gazebo_launch,
        robot_state_publisher,
        joint_state_publisher,
        robot_controller
    ])
```

## Domain Randomization Example

Implement domain randomization to improve sim-to-real transfer:

```python
#!/usr/bin/env python3

"""
Domain randomization for improved sim-to-real transfer
This example shows how to randomize simulation parameters
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
import random
import numpy as np

class DomainRandomizationNode(Node):
    def __init__(self):
        super().__init__('domain_randomization_node')
        
        # Publishers for different randomized parameters
        self.light_pub = self.create_publisher(Float32MultiArray, 'randomized_lighting', 10)
        self.friction_pub = self.create_publisher(Float32MultiArray, 'randomized_friction', 10)
        self.texture_pub = self.create_publisher(String, 'randomized_texture', 10)
        
        # Timer to periodically randomize parameters
        timer_period = 5.0  # Randomize every 5 seconds
        self.timer = self.create_timer(timer_period, self.randomize_environment)
        
        self.get_logger().info('Domain Randomization node initialized')

    def randomize_environment(self):
        """Randomize environment parameters"""
        # Randomize lighting conditions
        light_msg = Float32MultiArray()
        light_msg.data = [
            random.uniform(0.3, 2.0),  # Intensity
            random.uniform(-1, 1),     # X direction
            random.uniform(-1, 1),     # Y direction
            random.uniform(0, 1)       # Z direction
        ]
        self.light_pub.publish(light_msg)
        
        # Randomize friction coefficients
        friction_msg = Float32MultiArray()
        friction_msg.data = [
            random.uniform(0.1, 0.9),  # Static friction
            random.uniform(0.05, 0.5)  # Dynamic friction
        ]
        self.friction_pub.publish(friction_msg)
        
        # Randomize texture types
        textures = ['metal', 'wood', 'plastic', 'fabric', 'concrete', 'grass']
        texture_msg = String()
        texture_msg.data = random.choice(textures)
        self.texture_pub.publish(texture_msg)
        
        self.get_logger().info(
            f'Environment randomized - '
            f'Light: {light_msg.data}, '
            f'Friction: {friction_msg.data}, '
            f'Texture: {texture_msg.data}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DomainRandomizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Domain Randomization node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Monitoring

Monitor simulation performance to ensure realistic behavior:

```python
#!/usr/bin/env python3

"""
Simulation performance monitoring
Monitor simulation metrics to ensure realistic behavior
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import String
import time
import psutil

class SimulationMonitor(Node):
    def __init__(self):
        super().__init__('simulation_monitor')
        
        # Publishers for performance metrics
        self.cpu_pub = self.create_publisher(Float32, 'performance/cpu_usage', 10)
        self.memory_pub = self.create_publisher(Float32, 'performance/memory_usage', 10)
        self.sim_time_pub = self.create_publisher(Float32, 'performance/sim_time_factor', 10)
        self.status_pub = self.create_publisher(String, 'performance/status', 10)
        
        # Timer for monitoring
        timer_period = 1.0  # Monitor every second
        self.timer = self.create_timer(timer_period, self.monitor_performance)
        
        # Track real vs sim time
        self.real_start_time = time.time()
        self.last_sim_time = 0.0
        
        self.get_logger().info('Simulation Monitor node initialized')

    def monitor_performance(self):
        """Monitor simulation performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_msg = Float32()
        cpu_msg.data = float(cpu_percent)
        self.cpu_pub.publish(cpu_msg)
        
        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        memory_msg = Float32()
        memory_msg.data = float(memory_percent)
        self.memory_pub.publish(memory_msg)
        
        # Simulation time factor (how much faster simulation is running than real-time)
        # In a real implementation, this would compare ROS time to real time
        sim_time_factor = Float32()
        sim_time_factor.data = 1.0  # Placeholder - would be calculated in real implementation
        self.sim_time_pub.publish(sim_time_factor)
        
        # Overall status
        status_msg = String()
        if cpu_percent > 80 or memory_percent > 80:
            status_msg.data = "WARNING: High resource usage detected"
        else:
            status_msg.data = "OK: Performance within normal parameters"
        
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(
            f'Performance - CPU: {cpu_percent}%, '
            f'Memory: {memory_percent}%, '
            f'Status: {status_msg.data}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = SimulationMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Simulation Monitor node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Digital Twin Development

1. **Model Accuracy**: Ensure your digital twin accurately represents the physical system
2. **Validation**: Regularly validate simulation results against real-world data
3. **Scalability**: Design simulations that can scale to complex scenarios
4. **Reproducibility**: Make sure simulations are deterministic when needed
5. **Documentation**: Document all assumptions and parameters used in simulation

## Acceptance Scenarios

The intermediate simulation examples are properly implemented when:

**Scenario 1**: As a simulation developer, when I run the Gazebo simulation with the humanoid robot model, then the robot should move realistically with physics-based interactions.

**Scenario 2**: As a Unity developer, when I implement the Unity-based simulation, then the robot should respond to ROS commands and publish sensor data correctly.

**Scenario 3**: As a systems integrator, when I connect both Gazebo and Unity simulations to ROS 2, then they should operate in coordination with consistent state information.

## Next Steps

Once you have mastered these intermediate simulation examples, explore Module 3 on the AI-Robot Brain using NVIDIA Isaac, where you'll learn how to connect these simulation environments to AI systems for advanced robot intelligence.