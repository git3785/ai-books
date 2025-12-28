# VLA System Workflow Diagram

This diagram illustrates the data flow, cognitive planning, and ROS 2 action execution in the Vision-Language-Action system.

## High-Level Flow

```
[User Speech] 
      ↓
[Audio Input → Microphone]
      ↓
[Speech-to-Text → Whisper API]
      ↓
[Text Processing → NLP Pipeline with GPT-4]
      ↓
[Intent Recognition → Action Planning]
      ↓
[Action Mapping → ROS 2 Action Server]
      ↓
[Robot Execution → Physical Action]
      ↓
[Result Feedback → User]
```

## Detailed Workflow

1. **Audio Capture**: Robot microphone captures voice command
2. **Speech-to-Text**: Audio converted to text using OpenAI Whisper
3. **NLP Processing**: GPT-4 processes text to determine intent and parameters
4. **Action Planning**: Intent translated to ROS 2 action sequence
5. **Execution**: Actions executed on robot or simulation
6. **Status Feedback**: Execution status fed back to cognitive planning

## Cognitive Planning Process

```
Input: Natural Language Command
    ↓
Intent Classification
    ↓
Parameter Extraction
    ↓
Action Sequence Generation
    ↓
Safety Validation
    ↓
ROS 2 Action Execution
    ↓
Success/Failure Feedback
```

## ROS 2 Action Mapping

```
NLP Output → Action Type → ROS 2 Action Goal
    ↓
Action Server → Robot Hardware → Execution Result
    ↓
Feedback Loop → Cognitive Planning → Next Action
```