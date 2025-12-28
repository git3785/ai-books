---
sidebar_position: 1
title: "Advanced Digital Twin Concepts"
---

# Advanced Digital Twin Concepts: Gazebo and Unity Integration

This tutorial covers advanced concepts in digital twin technology for humanoid robotics, focusing on the integration of Gazebo and Unity simulation environments with ROS 2. You'll learn how to create sophisticated virtual environments that accurately represent physical robots and their operating conditions.

## Advanced Gazebo Simulation Techniques

### Creating Complex Robot Models (URDF/SDF)

Gazebo uses SDF (Simulation Description Format) to define robot models, though URDF (Unified Robot Description Format) models can also be used. Here's how to create a detailed humanoid robot model:

```xml
<?xml version="1.0" ?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Material definitions -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
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

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.4"/>
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
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Additional joints and links would continue here -->
</robot>
```

### Physics Parameters and Realism

To make your simulation more realistic, configure physics parameters carefully:

```xml
<!-- In your world file -->
<sdf version="1.6">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your robot would be included here -->
  </world>
</sdf>
```

### Advanced Sensor Simulation

Implement realistic sensor simulation with appropriate noise models:

```xml
<!-- Camera sensor with realistic parameters -->
<sensor name="camera_front" type="camera">
  <camera>
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_front_frame</frame_name>
    <min_depth>0.1</min_depth>
    <max_depth>10.0</max_depth>
  </plugin>
</sensor>

<!-- IMU sensor with noise model -->
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
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
    <frame_name>imu_link</frame_name>
    <topic>__default_topic__</topic>
  </plugin>
</sensor>
```

## Advanced Unity Simulation Techniques

### Setting up Unity for Robotics Simulation

Unity provides a powerful environment for creating high-fidelity digital twins. Here's how to set up Unity for robotics simulation:

1. **Install Unity Robotics Hub**: Download and install the Unity Robotics packages from the Unity Asset Store or GitHub.

2. **Configure ROS-TCP-Connector**: Set up the communication layer between Unity and ROS 2.

3. **Create Robot Models**: Import or create 3D models of your robot with appropriate physics properties.

### Unity Robotics Code Example

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class HumanoidRobotController : MonoBehaviour
{
    ROSConnection ros;
    string rosTopic = "unity_robot_status";

    // Robot joint transforms
    public Transform head;
    public Transform leftArm;
    public Transform rightArm;
    public Transform leftLeg;
    public Transform rightLeg;

    // Joint positions (for simulation)
    float headYaw = 0f;
    float headPitch = 0f;
    float leftArmYaw = 0f;
    float rightArmYaw = 0f;

    // Start is called before the first frame update
    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<StringMsg>(rosTopic);
    }

    // Update is called once per frame
    void Update()
    {
        // Example: Send robot status to ROS
        ros.Publish(rosTopic, new StringMsg("Robot is running in Unity simulation"));

        // Update joint positions based on ROS commands
        UpdateRobotJoints();
    }

    void UpdateRobotJoints()
    {
        // Update head position
        head.localRotation = Quaternion.Euler(headPitch, headYaw, 0);

        // Update arm positions
        leftArm.localRotation = Quaternion.Euler(0, leftArmYaw, 0);
        rightArm.localRotation = Quaternion.Euler(0, rightArmYaw, 0);
    }

    // Method to receive joint commands from ROS
    public void SetJointPosition(string jointName, float position)
    {
        switch(jointName)
        {
            case "head_yaw":
                headYaw = position;
                break;
            case "head_pitch":
                headPitch = position;
                break;
            case "left_arm_yaw":
                leftArmYaw = position;
                break;
            case "right_arm_yaw":
                rightArmYaw = position;
                break;
        }
    }
}
```

### ROS 2 Integration in Unity

To integrate Unity with ROS 2, you'll need to implement communication bridges:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class UnityCameraPublisher : MonoBehaviour
{
    public Camera unityCamera;
    ROSConnection ros;
    string cameraTopic = "unity_camera/image_raw";

    // Camera parameters
    int width = 640;
    int height = 480;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        StartCoroutine(PublishCameraFeed());
    }

    IEnumerator PublishCameraFeed()
    {
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = unityCamera.targetTexture;

        while (true)
        {
            // Capture image from Unity camera
            Texture2D imageTex = new Texture2D(width, height, TextureFormat.RGB24, false);
            imageTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            imageTex.Apply();

            // Convert to ROS Image message
            ImageMsg imageMsg = new ImageMsg();
            imageMsg.header = new std_msgs.HeaderMsg();
            imageMsg.header.stamp = new builtin_interfaces.TimeMsg();
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
            ros.Publish(cameraTopic, imageMsg);

            // Clean up
            Destroy(imageTex);

            // Wait for next frame (30 FPS)
            yield return new WaitForSeconds(1.0f / 30.0f);
        }

        RenderTexture.active = currentRT;
    }
}
```

## Sim-to-Real Transfer Considerations

### Domain Randomization

To improve sim-to-real transfer, implement domain randomization:

```python
#!/usr/bin/env python3

"""
Sim-to-real transfer considerations
Example of domain randomization in simulation
"""

import random
import rclpy
from rclpy.node import Node

class DomainRandomizationNode(Node):
    def __init__(self):
        super().__init__('domain_randomization_node')

        # Randomize environment parameters each time
        self.randomize_environment()

        # Timer to periodically update environment
        self.timer = self.create_timer(5.0, self.randomize_environment)

    def randomize_environment(self):
        """Randomize environmental parameters"""
        # Randomize lighting conditions
        light_intensity = random.uniform(0.5, 2.0)
        light_direction = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0, 1)]

        # Randomize object textures
        texture_variations = ['metal', 'wood', 'plastic', 'fabric']
        selected_texture = random.choice(texture_variations)

        # Randomize physics parameters
        friction_coeff = random.uniform(0.1, 0.9)
        restitution_coeff = random.uniform(0.0, 0.5)

        # Log the changes
        self.get_logger().info(
            f"Environment randomized - Light: {light_intensity}, "
            f"Texture: {selected_texture}, Friction: {friction_coeff}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = DomainRandomizationNode()

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

### Sensor Fusion in Simulation

Implement sensor fusion to combine data from multiple simulated sensors:

```python
#!/usr/bin/env python3

"""
Sensor fusion in simulation
Example combining camera, IMU, and LIDAR data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribers for different sensors
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)

        # Publisher for fused pose
        self.pose_pub = self.create_publisher(PoseStamped, 'fused_pose', 10)

        # Storage for sensor data
        self.latest_camera_data = None
        self.latest_imu_data = None
        self.latest_lidar_data = None

        # Timer for fusion
        self.timer = self.create_timer(0.1, self.fuse_sensors)

    def camera_callback(self, msg):
        """Process camera data"""
        self.latest_camera_data = msg
        # In a real implementation, extract features from the image
        self.get_logger().info("Received camera data")

    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu_data = msg
        # Extract orientation and angular velocity
        self.get_logger().info("Received IMU data")

    def lidar_callback(self, msg):
        """Process LIDAR data"""
        self.latest_lidar_data = msg
        # Process distance measurements
        self.get_logger().info("Received LIDAR data")

    def fuse_sensors(self):
        """Fuse sensor data using a simple complementary filter"""
        if self.latest_imu_data is not None:
            # Extract orientation from IMU
            orientation = self.latest_imu_data.orientation

            # Create fused pose message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.orientation = orientation

            # In a real implementation, you would combine data from all sensors
            # using more sophisticated fusion algorithms (Kalman filters, etc.)

            # Publish the fused pose
            self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

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

## Performance Optimization

### Gazebo Performance Tips

1. **Reduce visual complexity**: Use simplified models for real-time simulation
2. **Adjust physics parameters**: Balance accuracy with performance
3. **Limit update rates**: Set appropriate update rates for sensors
4. **Use multi-threading**: Enable Gazebo's multi-threaded physics

### Unity Performance Tips

1. **Optimize rendering**: Use Level of Detail (LOD) systems
2. **Reduce draw calls**: Batch similar objects together
3. **Use occlusion culling**: Hide objects not visible to cameras
4. **Optimize physics**: Adjust fixed timestep and solver iterations

## Best Practices for Digital Twin Development

1. **Model Accuracy**: Ensure your digital twin accurately represents the physical system
2. **Validation**: Regularly validate simulation results against real-world data
3. **Scalability**: Design simulations that can scale to complex scenarios
4. **Reproducibility**: Make sure simulations are deterministic when needed
5. **Documentation**: Document all assumptions and parameters used in simulation

## Acceptance Scenarios

The advanced digital twin concepts are properly implemented when:

**Scenario 1**: As a robotics engineer, when I create a Gazebo simulation of a humanoid robot, then it should accurately represent the physical robot's kinematics, dynamics, and sensor characteristics.

**Scenario 2**: As a simulation developer, when I implement Unity-based visualization for a robot system, then it should provide realistic visual rendering and integrate properly with ROS 2.

**Scenario 3**: As a researcher, when I develop sim-to-real transfer techniques, then the behaviors learned in simulation should transfer effectively to the physical robot with minimal performance degradation.

## Next Steps

Once you have mastered these advanced digital twin concepts, explore Module 3 on the AI-Robot Brain using NVIDIA Isaac, where you'll learn how to connect your simulation environments to AI systems for advanced robot intelligence.