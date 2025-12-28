#!/usr/bin/env python3

"""
Whisper Client for VLA Integration
This module handles audio recording and speech-to-text conversion using OpenAI Whisper API
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import openai
import pyaudio
import wave
import tempfile
import os
import threading
import time

class WhisperClientNode(Node):
    def __init__(self):
        super().__init__('whisper_client')
        
        # Publisher for transcribed text
        self.transcription_publisher = self.create_publisher(String, 'audio_transcription', 10)
        
        # Publisher for recording status
        self.recording_status_publisher = self.create_publisher(Bool, 'recording_status', 10)
        
        # Subscription to trigger recording
        self.trigger_subscription = self.create_subscription(
            Bool,
            'start_recording',
            self.trigger_callback,
            10
        )
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Audio settings
        self.chunk = 1024  # Record in chunks of 1024 samples
        self.format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 1  # Single channel
        self.rate = 44100  # 44.1kHz sampling rate
        self.record_seconds = 5  # Record for 5 seconds
        
        # Recording control
        self.is_recording = False
        
        self.get_logger().info("Whisper Client Node initialized")

    def trigger_callback(self, msg):
        """Callback to start recording when triggered"""
        if msg.data:  # If recording is triggered
            self.get_logger().info("Recording triggered")
            # Use threading to avoid blocking the ROS node
            recording_thread = threading.Thread(target=self.start_recording)
            recording_thread.start()

    def start_recording(self):
        """Start audio recording and processing"""
        if self.is_recording:
            self.get_logger().warn("Recording already in progress")
            return
            
        self.is_recording = True
        
        # Publish recording status
        status_msg = Bool()
        status_msg.data = True
        self.recording_status_publisher.publish(status_msg)
        
        self.get_logger().info("Starting audio recording...")
        frames = []
        
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.get_logger().info("Recording...")
            
            # Record audio
            for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
                data = stream.read(self.chunk)
                frames.append(data)
                
            self.get_logger().info("Finished recording")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                wf = wave.open(temp_file.name, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                # Transcribe audio
                transcription = self.transcribe_audio(temp_file.name)
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                if transcription:
                    # Publish transcription
                    msg = String()
                    msg.data = transcription
                    self.transcription_publisher.publish(msg)
                    self.get_logger().info(f'Published transcription: {transcription}')
                else:
                    self.get_logger().error('Transcription failed')
        
        except Exception as e:
            self.get_logger().error(f'Error during recording: {e}')
        finally:
            self.is_recording = False
            # Publish recording status
            status_msg = Bool()
            status_msg.data = False
            self.recording_status_publisher.publish(status_msg)

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                return transcript.text
        except Exception as e:
            self.get_logger().error(f'Error transcribing audio: {e}')
            return None

    def transcribe_audio_offline(self, audio_file_path):
        """
        Edge-optimized transcription using local model
        This is a placeholder - in a real implementation, you would use
        a local ASR model like faster-whisper or similar
        """
        try:
            # Placeholder for local transcription
            # In a real implementation, this would use a local model
            # like faster-whisper: https://github.com/guillaumekln/faster-whisper
            """
            from faster_whisper import WhisperModel

            model_size = "base"  # Choose based on device capabilities
            model = WhisperModel(model_size, device="cuda", compute_type="float16")

            segments, info = model.transcribe(audio_file_path, beam_size=5)
            transcription = " ".join([segment.text for segment in segments])
            return transcription
            """
            # For now, return a mock response to demonstrate the concept
            self.get_logger().info(f"Would transcribe {audio_file_path} using local model")
            return "Local transcription not implemented in this example"
        except Exception as e:
            self.get_logger().error(f'Error in local transcription: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    whisper_client = WhisperClientNode()
    
    try:
        rclpy.spin(whisper_client)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()