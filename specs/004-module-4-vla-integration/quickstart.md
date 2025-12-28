# Quickstart Guide: Vision-Language-Action (VLA) Integration

## Overview
This guide provides a rapid introduction to implementing Vision-Language-Action integration for humanoid robots using OpenAI Whisper, GPT-4, and ROS 2.

## Prerequisites
- ROS 2 Humble Hawksbill installed
- OpenAI API account with Whisper and GPT-4 access
- NVIDIA Isaac ROS packages
- Jetson Orin Nano/NX (for hardware deployment) or simulation environment (Gazebo/Unity)

## Setup Steps

### 1. Environment Setup
```bash
# Install ROS 2 Humble dependencies
sudo apt update
sudo apt install ros-humble-desktop ros-humble-ros-base
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Install OpenAI Python package
pip3 install openai

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Create ROS 2 Workspace
```bash
# Create workspace
mkdir -p ~/vla_ws/src
cd ~/vla_ws

# Source ROS 2
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### 3. VLA Integration Package
```bash
# Navigate to source directory
cd ~/vla_ws/src

# Create VLA package
ros2 pkg create --build-type ament_python vla_integration
cd vla_integration
```

### 4. Basic VLA Node Structure
Create the main VLA integration node:

**vla_integration/vla_integration/vla_node.py**
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import os

class VLANode(Node):
    def __init__(self):
        super().__init__('vla_node')
        
        # Publisher for robot actions
        self.action_publisher = self.create_publisher(String, 'robot_actions', 10)
        
        # Subscriber for audio transcription
        self.audio_subscriber = self.create_subscription(
            String,
            'audio_transcription',
            self.audio_callback,
            10
        )
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.get_logger().info("VLA Node initialized")

    def audio_callback(self, msg):
        try:
            # Process the transcribed text with GPT-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a robot command interpreter. Convert natural language commands to simple action sequences. Respond with just the action name and parameters."},
                    {"role": "user", "content": msg.data}
                ]
            )
            
            # Publish the robot action
            action_msg = String()
            action_msg.data = response.choices[0].message.content
            self.action_publisher.publish(action_msg)
            
            self.get_logger().info(f'Published action: {action_msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLANode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 5. Whisper Integration Node
**vla_integration/vla_integration/whisper_node.py**
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import pyaudio
import wave
import os

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')
        
        # Publisher for transcribed text
        self.transcription_publisher = self.create_publisher(String, 'audio_transcription', 10)
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 5
        self.wav_filename = "temp_audio.wav"
        
        self.get_logger().info("Whisper Node initialized")

    def record_audio(self):
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info("Recording...")
        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        self.get_logger().info("Finished recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save audio to WAV file
        wf = wave.open(self.wav_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def transcribe_audio(self):
        try:
            with open(self.wav_filename, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                return transcript.text
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')
            return ""

    def process_audio(self):
        self.record_audio()
        transcription = self.transcribe_audio()
        
        if transcription:
            msg = String()
            msg.data = transcription
            self.transcription_publisher.publish(msg)
            self.get_logger().info(f'Published transcription: {transcription}')

def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperNode()
    
    # Process audio and publish transcription
    whisper_node.process_audio()
    
    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 6. Launch File
**vla_integration/launch/vla_integration.launch.py**
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vla_integration',
            executable='vla_node',
            name='vla_node',
            output='screen'
        ),
        Node(
            package='vla_integration',
            executable='whisper_node', 
            name='whisper_node',
            output='screen'
        )
    ])
```

### 7. Run the Integration
```bash
# Build the workspace
cd ~/vla_ws
source install/setup.bash
colcon build --packages-select vla_integration

# Run the nodes
source install/setup.bash
ros2 launch vla_integration vla_integration.launch.py
```

## Testing the Integration
1. Say a simple command to the robot (e.g., "Move forward")
2. The Whisper node captures audio and transcribes it
3. The VLA node processes the transcription with GPT-4
4. The resulting action is published to the robot_actions topic
5. The action can then be processed by the robot's action server

## Next Steps
- Add vision integration for multi-modal perception
- Implement cognitive planning for complex tasks
- Deploy on Jetson edge hardware
- Integrate with humanoid robot controllers