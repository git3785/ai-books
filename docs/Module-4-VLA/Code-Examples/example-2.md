---
sidebar_label: 'Example 2'
title: 'Example 2'
---

# Code Example 2

This is another sample code example for the VLA integration module.

```python
# Sample Python code for ROS 2 action mapping
import rclpy
from rclpy.action import ActionClient
from std_msgs.msg import String

class VLAActionClient:
    def __init__(self):
        self.node = rclpy.create_node('vla_action_client')
        self._action_client = ActionClient(self.node, CustomAction, 'perform_vla_action')

    def send_goal(self, command):
        goal_msg = CustomAction.Goal()
        goal_msg.command = command
        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)
```