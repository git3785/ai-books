---
sidebar_position: 2
title: "NLP Pipeline Tutorial"
---

# NLP Pipeline Tutorial

This tutorial covers creating a natural language processing pipeline that translates human commands into executable ROS 2 actions.

## Overview

After converting speech to text using Whisper, we need to process the natural language to understand the user's intent and convert it into actionable commands for the robot. This tutorial covers how to implement an NLP pipeline using GPT-4 for cognitive planning and intent recognition.

## Prerequisites

- Completed Whisper Integration tutorial
- OpenAI API access with GPT-4
- ROS 2 Humble installation

## Setting up the NLP Pipeline

The NLP pipeline involves several steps:

1. Processing the transcribed text
2. Identifying the user's intent
3. Extracting relevant parameters
4. Mapping to ROS 2 action sequences

## Implementation Steps

First, let's set up the basic NLP processing with GPT-4:

```python
import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def process_natural_language_command(transcription):
    # Define the system prompt for GPT-4
    system_prompt = """
    You are a robot command interpreter. Convert natural language commands to structured action objects.
    The possible actions are: move_to, manipulate_object, speak, perception_task.
    Respond with a JSON object containing the action type and parameters.
    Example response: {"action_type": "move_to", "parameters": {"x": 1.0, "y": 2.0, "z": 0.0}}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcription}
        ],
        temperature=0.1  # Lower temperature for more consistent outputs
    )
    
    try:
        # Parse the JSON response
        action_data = json.loads(response.choices[0].message.content)
        return action_data
    except json.JSONDecodeError:
        # Handle cases where GPT-4 doesn't return valid JSON
        print("GPT-4 response wasn't valid JSON:", response.choices[0].message.content)
        return None
```

## Integration with ROS 2

To integrate the NLP pipeline with ROS 2, we'll create a node that:

1. Subscribes to the audio transcription topic
2. Processes the transcription with our NLP pipeline
3. Publishes the resulting action to an action sequence topic

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import os
import json

class NLPProcessorNode(Node):
    def __init__(self):
        super().__init__('nlp_processor')
        
        # Subscribe to audio transcription
        self.subscription = self.create_subscription(
            String,
            'audio_transcription',
            self.transcription_callback,
            10
        )
        
        # Publish action sequences
        self.action_publisher = self.create_publisher(String, 'action_sequence', 10)
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.get_logger().info("NLP Processor Node initialized")

    def transcription_callback(self, msg):
        transcription = msg.data
        
        # Process with GPT-4
        action_data = self.process_command(transcription)
        
        if action_data:
            # Publish the action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_data)
            self.action_publisher.publish(action_msg)
            self.get_logger().info(f'Published action: {action_data}')

    def process_command(self, transcription):
        system_prompt = """
        You are a robot command interpreter. Convert natural language commands to structured action objects.
        The possible actions are: move_to, manipulate_object, speak, perception_task.
        Respond with a JSON object containing the action type and parameters.
        Example response: {"action_type": "move_to", "parameters": {"x": 1.0, "y": 2.0, "z": 0.0}}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    nlp_node = NLPProcessorNode()
    
    try:
        rclpy.spin(nlp_node)
    except KeyboardInterrupt:
        pass
    finally:
        nlp_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Cognitive Planning for Humanoid Robots

Cognitive planning involves creating systems that can reason, plan, and make decisions like a human would. For humanoid robots, this means integrating perception, reasoning, and action in a coordinated way.

### Multi-Step Planning

For complex tasks, the robot needs to break down high-level commands into sequences of simpler actions:

```python
def plan_complex_task(nlp_result):
    """
    Plan a multi-step task based on NLP result
    Example: "Go to the kitchen, pick up the red cup, and bring it to me"
    Would become: [navigate_to_kitchen, locate_red_cup, grasp_cup, return_to_user]
    """
    if nlp_result.get('action_type') == 'complex_task':
        # Extract subtasks
        subtasks = [
            {'action_type': 'navigate', 'parameters': {'location': 'kitchen'}},
            {'action_type': 'locate_object', 'parameters': {'object': 'red cup'}},
            {'action_type': 'grasp', 'parameters': {'object': 'red cup'}},
            {'action_type': 'navigate', 'parameters': {'location': 'user'}},
        ]
        return {'action_type': 'task_sequence', 'parameters': {'subtasks': subtasks}}

    return nlp_result
```

### Context and Memory

For cognitive planning, robots need to maintain context across multiple interactions:

```python
class ContextManager:
    def __init__(self):
        self.context = {
            'current_location': (0, 0, 0),
            'last_action': None,
            'conversation_history': [],
            'object_locations': {},
            'user_preferences': {}
        }

    def update_context(self, action_result):
        """Update the robot's context based on action results"""
        if 'location' in action_result:
            self.context['current_location'] = action_result['location']

        self.context['last_action'] = action_result

    def get_context_aware_prompt(self, user_command):
        """Generate a prompt that includes the current context"""
        context_str = f"Robot location: {self.context['current_location']}"
        context_str += f"Last action: {self.context['last_action']}"
        # Include other relevant context information

        return f"{context_str}\n\nUser command: {user_command}"
```

### Reasoning and Decision Making

Implement reasoning capabilities to handle unexpected situations:

```python
def make_decision(perception_data, planned_action, context):
    """
    Make decisions based on perception, planned actions, and context
    """
    # Check if planned action is still valid given current perception
    if planned_action['action_type'] == 'navigate':
        # Check for obstacles in the planned path
        if 'obstacle_detected' in perception_data:
            # Modify action to navigate around obstacle
            return {
                'action_type': 'navigate_around_obstacle',
                'parameters': {
                    'original_destination': planned_action['parameters']['destination'],
                    'obstacle_position': perception_data['obstacle_position']
                }
            }

    # If no issues, return original planned action
    return planned_action
```

## Advanced NLP Techniques

For more complex interactions, you can enhance the NLP pipeline with:

- Context awareness (using conversation history)
- Entity recognition for specific robot capabilities
- Intent classification to improve accuracy
- Error handling for ambiguous commands

## Best Practices

- Always validate GPT-4 output before executing robot actions
- Implement safety checks before executing commands
- Consider the robot's current state when processing commands
- Log all commands for debugging and safety analysis

## Acceptance Scenarios

The NLP pipeline is working correctly when:

**Scenario 1**: As a user, when I give a natural language command like "Go to the kitchen and grab the red cup", then the NLP pipeline should correctly identify the intent as a sequence of actions (navigation + manipulation) with appropriate parameters.

**Scenario 2**: As a developer, when I test the pipeline with ambiguous commands, then the system should either provide a reasonable interpretation or request clarification rather than executing incorrect actions.

**Scenario 3**: As a researcher, when I evaluate the cognitive planning capabilities, then the system should demonstrate context awareness and multi-step planning based on the robot's current state and environment.

## Next Steps

Once you have the NLP pipeline working, proceed to the [ROS 2 Action Mapping tutorial](ros2-action-mapping.md) to execute the planned actions on your robot.

For implementing the complete VLA system, also see:
- [Whisper Integration](whisper-integration.md) - Provides the audio input for the NLP pipeline
- [Edge Deployment](edge-deployment.md) - Deploys the complete system on Jetson hardware
- [Multi-Modal Perception](multi-modal-perception.md) - Integrates vision, language, and action for enhanced capabilities