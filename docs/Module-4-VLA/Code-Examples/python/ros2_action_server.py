#!/usr/bin/env python3

"""
ROS 2 Action Server for VLA Integration
This module implements action servers that execute commands from the NLP pipeline
"""

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import math

class VLAActionServer(Node):
    def __init__(self):
        super().__init__('vla_action_server')
        
        # Subscribe to action sequences from NLP processor
        self.action_subscription = self.create_subscription(
            String,
            'action_sequence',
            self.action_sequence_callback,
            10
        )
        
        self.get_logger().info("VLA Action Server initialized")

    def action_sequence_callback(self, msg):
        """Process incoming action sequences"""
        try:
            action_data = json.loads(msg.data)
            action_type = action_data.get('action_type')
            
            self.get_logger().info(f"Received action: {action_type}")
            
            # Execute the action based on its type
            if action_type == 'move_to':
                self.execute_move_to(action_data)
            elif action_type == 'manipulate_object':
                self.execute_manipulate_object(action_data)
            elif action_type == 'speak':
                self.execute_speak(action_data)
            elif action_type == 'perception_task':
                self.execute_perception_task(action_data)
            else:
                self.get_logger().error(f"Unknown action type: {action_type}")
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in action sequence")
        except Exception as e:
            self.get_logger().error(f"Error processing action sequence: {e}")

    def execute_move_to(self, action_data):
        """Execute move_to action"""
        params = action_data.get('parameters', {})
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        z = params.get('z', 0.0)

        self.get_logger().info(f"Moving to position: ({x}, {y}, {z})")

        # In a real implementation, this would interface with the robot's navigation system
        # For edge optimization, check resource usage before executing
        if self.should_delay_execution():
            self.get_logger().warn("High resource usage, delaying execution")
            time.sleep(1)  # Brief delay under high load

        # Simulate movement execution
        # In a real implementation, this would interface with the robot's navigation system
        time.sleep(2)  # Simulate movement time

        self.get_logger().info("Move completed")

    def should_delay_execution(self):
        """Check if resources are constrained and execution should be delayed"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Check memory usage
            memory_percent = psutil.virtual_memory().percent

            # High resource usage threshold
            return cpu_percent > 80 or memory_percent > 80
        except:
            # If we can't check resources, assume it's OK to proceed
            return False

    def execute_manipulate_object(self, action_data):
        """Execute manipulate_object action"""
        params = action_data.get('parameters', {})
        object_id = params.get('object_id', '')
        action_type = params.get('action_type', 'grab')
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        z = params.get('z', 0.0)
        
        self.get_logger().info(f"Manipulating object {object_id} with action {action_type} at ({x}, {y}, {z})")
        
        # Simulate manipulation
        # In a real implementation, this would control the robot's manipulator
        time.sleep(1.5)  # Simulate manipulation time
        
        self.get_logger().info("Manipulation completed")

    def execute_speak(self, action_data):
        """Execute speak action"""
        params = action_data.get('parameters', {})
        text = params.get('text_to_speak', '')
        
        self.get_logger().info(f"Speaking: {text}")
        
        # Simulate speaking
        # In a real implementation, this would use text-to-speech
        time.sleep(len(text) * 0.1)  # Simulate time based on text length
        
        self.get_logger().info("Speech completed")

    def execute_perception_task(self, action_data):
        """Execute perception_task action"""
        params = action_data.get('parameters', {})
        task_type = params.get('task_type', 'object_detection')
        
        self.get_logger().info(f"Performing perception task: {task_type}")
        
        # Simulate perception task
        # In a real implementation, this would interface with perception systems
        time.sleep(1)  # Simulate perception time
        
        self.get_logger().info("Perception task completed")

def main(args=None):
    rclpy.init(args=args)
    action_server = VLAActionServer()
    
    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()