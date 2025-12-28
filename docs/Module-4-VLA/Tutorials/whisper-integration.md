---
sidebar_position: 1
title: "Whisper Integration Tutorial"
---

# Whisper Integration Tutorial

This tutorial covers integrating OpenAI Whisper with humanoid robots for voice-command control.

## Overview

OpenAI's Whisper is a state-of-the-art automatic speech recognition (ASR) system. In this tutorial, we'll cover how to integrate Whisper into your humanoid robot system to enable voice-command functionality.

## Prerequisites

- OpenAI API account with Whisper access
- ROS 2 Humble installation
- Audio input device connected to your robot

## Setting up Whisper API

First, you'll need to obtain an API key from OpenAI and set it up in your environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Implementation Steps

1. Record audio from the robot's microphone
2. Send the audio to the Whisper API for transcription
3. Process the transcribed text for further natural language processing

```python
import openai
import os
import pyaudio
import wave

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

def record_audio(filename="temp_audio.wav", record_seconds=5):
    # Audio recording implementation
    pass

def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript.text

# Example usage
audio_file = record_audio()
transcription = transcribe_audio(audio_file)
print(f"Transcription: {transcription}")
```

## Integration with ROS 2

To integrate Whisper with ROS 2, you'll need to create a node that:

1. Records audio when triggered
2. Sends the audio to Whisper API
3. Publishes the transcription to a topic for other nodes to process

Here's a basic structure for a Whisper node:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openai
import os

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')
        self.publisher = self.create_publisher(String, 'audio_transcription', 10)
        # Additional initialization code
        self.get_logger().info("Whisper Node initialized")

def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperNode()
    
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

## Best Practices

- Handle network failures gracefully
- Implement audio preprocessing to improve transcription quality
- Consider privacy implications when processing voice data
- Use appropriate audio formats for optimal Whisper API performance

## Acceptance Scenarios

The Whisper integration is working correctly when:

**Scenario 1**: As a user with a humanoid robot, when I speak a clear command like "Move forward 2 meters", then the robot's Whisper client should capture the audio, convert it to text, and publish the transcription for further processing.

**Scenario 2**: As a developer, when I test the Whisper node with various audio inputs, then it should consistently return accurate transcriptions with acceptable confidence levels (>0.8).

**Scenario 3**: As a researcher, when I evaluate the Whisper integration in a noisy environment, then the system should either provide accurate transcription or clearly indicate low confidence to trigger alternative processing.

## Next Steps

Once you have Whisper integration working, you can proceed to the [NLP Pipeline tutorial](nlp-pipeline.md) to process the transcribed text and generate robot actions.

For implementing the complete VLA system, also see:
- [ROS 2 Action Mapping](ros2-action-mapping.md) - Maps NLP outputs to ROS 2 actions
- [Edge Deployment](edge-deployment.md) - Deploys the complete system on Jetson hardware
- [Multi-Modal Perception](multi-modal-perception.md) - Integrates vision, language, and action