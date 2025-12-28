---
sidebar_position: 3
title: "ROS 2 Action Mapping Tutorial"
---

# ROS 2 Action Mapping Tutorial

This tutorial covers mapping natural language commands to ROS 2 action sequences for humanoid robot execution.

## Overview

After processing natural language through our NLP pipeline, we need to map the resulting commands to ROS 2 actions that our humanoid robot can execute. This involves setting up action servers and clients that can handle the commands generated from natural language processing.

## Prerequisites

- Completed NLP Pipeline tutorial
- ROS 2 Humble installation
- Basic understanding of ROS 2 actions

## Understanding ROS 2 Actions

ROS 2 actions are a communication pattern that allows for long-running tasks with feedback. They consist of:
- Goal: The request sent to the action server
- Feedback: Periodic updates on the action progress
- Result: The final outcome of the action

## Creating Action Definition

First, we'll define a generic action message for our humanoid robot:

```
# MoveTo.action
geometry_msgs/Pose target_pose
---
geometry_msgs/Pose final_pose
string status
---
string feedback_message
```

```
# ManipulateObject.action
string object_id
string action_type # grab, release, move
geometry_msgs/Pose target_pose
---
bool success
string error_message
---
float64 progress
string feedback_message
```

## Implementation Steps

1. Create the action server nodes in your robot's controller
2. Implement action clients that can receive commands from the NLP pipeline
3. Map action types from the NLP output to specific robot capabilities

### Action Server Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time

# Assuming custom action definitions are in your package
from your_robot_msgs.action import MoveTo, ManipulateObject

class MoveToActionServer(Node):
    def __init__(self):
        super().__init__('move_to_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            MoveTo,
            'move_to',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.get_logger().info("MoveTo Action Server initialized")

    def goal_callback(self, goal_request):
        """Accept or reject a goal."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        feedback_msg = MoveTo.Feedback()
        result = MoveTo.Result()
        
        # Simulate movement to target position
        target_pose = goal_request.target_pose
        
        # In a real implementation, this would control the robot's motors
        # For simulation, we'll just use sleep and update feedback
        for i in range(0, 101, 10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result.status = "canceled"
                return result
            
            # Update feedback
            feedback_msg.feedback_message = f"Moving to position: {i}% complete"
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate movement
            time.sleep(0.1)
        
        # At this point, the robot has reached the target
        result.final_pose = target_pose
        result.status = "success"
        
        goal_handle.succeed()
        self.get_logger().info('Goal succeeded')
        
        return result

def main(args=None):
    rclpy.init(args=args)

    move_to_server = MoveToActionServer()

    executor = MultiThreadedExecutor()
    executor.add_node(move_to_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        move_to_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
import json

from your_robot_msgs.action import MoveTo

class ActionMapperNode(Node):
    def __init__(self):
        super().__init__('action_mapper')
        
        # Subscribe to NLP pipeline output
        self.subscription = self.create_subscription(
            String,
            'action_sequence',
            self.action_callback,
            10
        )
        
        # Create action client for MoveTo action
        self.move_to_client = ActionClient(self, MoveTo, 'move_to')
        
        self.get_logger().info("Action Mapper Node initialized")

    def action_callback(self, msg):
        action_data_str = msg.data
        try:
            action_data = json.loads(action_data_str)
            action_type = action_data.get('action_type')
            
            if action_type == 'move_to':
                self.send_move_to_goal(action_data)
            elif action_type == 'manipulate_object':
                self.send_manipulate_goal(action_data)
            # Add other action types as needed
        except Exception as e:
            self.get_logger().error(f'Error parsing action data: {e}')

    def send_move_to_goal(self, action_data):
        # Wait for action server
        self.move_to_client.wait_for_server()
        
        # Extract parameters
        params = action_data.get('parameters', {})
        target_x = params.get('x', 0.0)
        target_y = params.get('y', 0.0)
        target_z = params.get('z', 0.0)
        
        # Create goal
        goal_msg = MoveTo.Goal()
        goal_msg.target_pose.position.x = target_x
        goal_msg.target_pose.position.y = target_y
        goal_msg.target_pose.position.z = target_z
        
        # Send goal
        self.move_to_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    action_mapper = ActionMapperNode()
    
    try:
        rclpy.spin(action_mapper)
    except KeyboardInterrupt:
        pass
    finally:
        action_mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with NLP Pipeline

To connect the action mapping with the NLP pipeline, you'll need to:

1. Ensure the NLP pipeline outputs action format compatible with your ROS 2 action definitions
2. Create a mapping layer that translates NLP outputs to ROS 2 action goals
3. Handle action execution feedback and results

## Best Practices

- Always implement proper error handling for action failures
- Provide feedback during long-running actions
- Validate action parameters before execution
- Implement safety checks to prevent dangerous robot movements

## Technical Claims Validation

The following technical claims in this tutorial have been validated against authoritative sources:

**Claim**: ROS 2 actions are ideal for long-running tasks with feedback.
**Validation**: This is confirmed in the official ROS 2 documentation on actions, which states that actions are designed for long-running goals that require status updates and feedback. (ROS 2 Documentation. (2023). Action Architecture. https://docs.ros.org/en/humble/Concepts/About-Actions.html)

**Claim**: Action servers and clients communicate asynchronously.
**Validation**: This is supported by the ROS 2 design principles and is documented in the official ROS 2 documentation on action architecture.

**Claim**: Using ReentrantCallbackGroup allows for multiple simultaneous operations.
**Validation**: The ROS 2 documentation confirms that ReentrantCallbackGroup allows callbacks to be executed in parallel, which is essential for action servers handling multiple goals simultaneously.

## Acceptance Scenarios

The ROS 2 action mapping is working correctly when:

**Scenario 1**: As a user, when I issue a command that should cause robot movement, then the corresponding ROS 2 action should be sent to the robot's navigation system and executed with appropriate feedback.

**Scenario 2**: As a developer, when I send various action commands to the system, then each should be properly mapped to the correct ROS 2 action server with appropriate parameters and error handling.

**Scenario 3**: As a researcher, when I evaluate the action execution pipeline, then the system should provide real-time feedback during long-running actions and proper result reporting upon completion.

## Next Steps

Once you have action mapping working, proceed to the [Edge Deployment tutorial](edge-deployment.md) to deploy your VLA system on Jetson hardware.

For implementing the complete VLA system, also see:
- [Whisper Integration](whisper-integration.md) - Provides audio input processed through this action mapping
- [NLP Pipeline](nlp-pipeline.md) - Generates the commands that this tutorial maps to actions
- [Multi-Modal Perception](multi-modal-perception.md) - Integrates multiple sensory inputs for better action planning