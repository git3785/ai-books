#!/usr/bin/env python3

"""
NLP Processor for VLA Integration
This module processes natural language commands and converts them to ROS 2 action sequences
"""

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
        
        # Publisher for action sequences
        self.action_publisher = self.create_publisher(String, 'action_sequence', 10)
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        self.get_logger().info("NLP Processor Node initialized")

    def transcription_callback(self, msg):
        """Process transcribed text and generate action sequence"""
        transcription = msg.data
        self.get_logger().info(f"Processing transcription: {transcription}")
        
        # Process with GPT-4
        action_data = self.process_command(transcription)
        
        if action_data:
            # Publish the action sequence
            action_msg = String()
            action_msg.data = json.dumps(action_data)
            self.action_publisher.publish(action_msg)
            self.get_logger().info(f'Published action: {action_data}')
        else:
            self.get_logger().error('Failed to process command')

    def process_command(self, transcription):
        """
        Process the natural language command using GPT-4
        Returns a structured action object
        """
        system_prompt = """You are a robot command interpreter. Convert natural language commands to structured action objects.
        The possible actions are: move_to, manipulate_object, speak, perception_task.
        Respond with a JSON object containing the action type and parameters.
        Example response: {"action_type": "move_to", "parameters": {"x": 1.0, "y": 2.0, "z": 0.0}}
        Keep responses concise and focused only on the JSON object."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ],
                temperature=0.1  # Lower temperature for more consistent outputs
            )

            # Extract the content from the response
            content = response.choices[0].message.content.strip()

            # Remove any markdown formatting if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```

            # Parse the JSON response
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON decode error: {e}')
            self.get_logger().error(f'Response content: {response.choices[0].message.content}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            return None

    def process_command_offline(self, transcription):
        """
        Edge-optimized NLP processing using local model
        This is a placeholder - in a real implementation, you would use
        a local LLM like a quantized version of Llama or similar
        """
        try:
            # Placeholder for local LLM processing
            # In a real implementation, this would use a local model
            # such as a quantized Llama model with an inference library
            """
            from transformers import pipeline

            # Load a quantized model for edge deployment
            classifier = pipeline("text-classification",
                                model="your_quantized_model_path",
                                device=0)  # Use GPU if available

            # Process the transcription
            result = classifier(transcription)
            # Convert result to appropriate action format
            """
            # For now, return a mock response to demonstrate the concept
            self.get_logger().info(f"Would process {transcription} using local model")
            return {"action_type": "speak", "parameters": {"text_to_speak": f"I heard you say: {transcription}"}}
        except Exception as e:
            self.get_logger().error(f'Error in local NLP processing: {e}')
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