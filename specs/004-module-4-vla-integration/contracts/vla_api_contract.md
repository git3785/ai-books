# API Contract: Vision-Language-Action (VLA) System

## Overview
This document defines the API contracts for the Vision-Language-Action integration module, focusing on the interfaces between speech processing, language understanding, and robot action execution.

## Speech-to-Text Service (Whisper Integration)

### `/process_audio` POST
Process audio input and return transcribed text

**Request**:
- Content-Type: `audio/wav` or `audio/mpeg`
- Body: Raw audio data

**Response**:
```
{
  "transcription": "string",
  "confidence": float,
  "processing_time_ms": int
}
```

### `/health` GET
Check if the speech-to-text service is available

**Response**:
```
{
  "status": "healthy" | "unhealthy",
  "timestamp": "ISO 8601 datetime"
}
```

## Natural Language Processing Service (GPT Integration)

### `/process_command` POST
Process natural language command and return action sequence

**Request**:
```
{
  "command": "string",
  "robot_context": {
    "location": "string",
    "available_actions": ["string"]
  }
}
```

**Response**:
```
{
  "intent": "string",
  "action_sequence": [
    {
      "action_type": "string",
      "parameters": {}
    }
  ],
  "confidence": float
}
```

## ROS 2 Action Server Interface

### Action: `/execute_action_sequence` 
Execute a sequence of actions on the humanoid robot

**Goal**:
```
{
  "actions": [
    {
      "type": "move_to | manipulate_object | speak | perception_task",
      "parameters": {
        "target_position": { "x": float, "y": float, "z": float } | 
                         { "object_id": string },
        "gripper_position": float,
        "text_to_speak": string
      }
    }
  ],
  "timeout_seconds": int
}
```

**Result**:
```
{
  "completed_actions": int,
  "total_actions": int,
  "error_message": "string" (optional)
}
```

## Multi-Modal Perception Service

### `/process_sensory_input` POST
Process combined sensory input (vision, audio, etc.) for decision making

**Request**:
```
{
  "audio_input": "string",
  "visual_data_ref": "string",
  "current_state": {
    "battery_level": float,
    "location": { "x": float, "y": float, "z": float },
    "gripper_status": "open | closed | unknown"
  }
}
```

**Response**:
```
{
  "cognitive_plan": {
    "primary_action": "string",
    "secondary_actions": ["string"],
    "confidence": float
  },
  "perception_data": {
    "recognized_objects": [{"id": "string", "type": "string", "position": {}}]
  }
}
```

## Edge Deployment Management

### `/deployment_status` GET
Get current status of the VLA system on edge hardware

**Response**:
```
{
  "system_status": "idle | processing | executing | error",
  "compute_usage": {
    "cpu_percent": float,
    "gpu_percent": float,
    "memory_percent": float
  },
  "active_services": ["string"],
  "last_error": "string" (optional)
}
```

### `/deployment_config` PUT
Update configuration for edge deployment

**Request**:
```
{
  "compute_allocation": {
    "cpu_cores": int,
    "gpu_memory_mb": int
  },
  "sensitivity_settings": {
    "audio_threshold": float,
    "vision_confidence_threshold": float
  }
}
```

**Response**:
```
{
  "status": "updated | error",
  "message": "string"
}
```