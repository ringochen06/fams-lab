# Student Behavior Detection System

## Core Detection Features

### 1. Face Detection

- **MediaPipe Face Mesh**: 468 facial landmarks
- **Extracts**: Nose position (head tracking), eye aspect ratio (EAR), facial contours

### 2. Pose Detection

- **MediaPipe Pose**: 33 body keypoints
- **Extracts**: Shoulder/hip centers, body angle (forward lean = engaged, backward = bored)

### 3. Hand Detection

- **MediaPipe Hands**: 21 keypoints per hand
- **Extracts**: Wrist positions for motion tracking

## Behavior Analysis

### 1. Nodding Detection (`NODDING_TOO_MUCH`)

- Tracks head vertical movement (nose Y-coordinate)
- Detects complete nod cycles (valley-peak-valley patterns)
- **Triggers when**: Frequency > 2.0 nods/sec, ≥3 cycles, variance > 30

### 2. Staring Detection (`NOT_UNDERSTANDING`)

- Detects prolonged gaze without note-taking
- **Conditions**: Stable head (variance < 50), eyes open (EAR > 0.3), hands still (variance < 100)

### 3. Note-Taking Detection (`TAKING_NOTES`)

- Tracks hand movement variance
- **Triggers when**: Hand variance > 200 (significant motion)

### 4. Posture Analysis

- **ENGAGED**: Forward lean (angle < -10°)
- **BORED**: Backward lean (angle > 10°)

### 5. Distraction Detection (`DISTRACTED`)

- Triggers when face is not detected

## Integrated AI Tutor (`integrated_tutor.py`)

### Board Analysis

- Press 'b' to capture and analyze board content
- Uses Gemini Vision API to extract text, formulas, diagrams
- Generates content summary

### Personalized Responses

Generates adaptive responses based on detected behaviors:

- **Nodding too much**: "You understand well, let's move faster"
- **Not understanding**: "Let me explain differently" + board context
- **Taking notes**: "Great! Keep taking notes"
- **Bored**: "Let me try a different approach"
- **Engaged**: "Continue at this pace"
- **Distracted**: "Let's refocus"

## Output Format

```python
{
    "face_detected": bool,
    "pose_detected": bool,
    "hands_detected": bool,
    "states": [StudentState],
    "recommendations": [str],
    "metrics": {
        "nod_frequency": float,
        "is_staring": bool,
        "is_taking_notes": bool,
        "posture": str
    }
}
```

## Usage

### Basic Mode

```bash
python behavior_detector.py
```

- Real-time behavior detection
- Press 'q' to quit

### Integrated Mode

```bash
python integrated_tutor.py
```

- Behavior detection + board analysis
- Press 'b' to analyze board
- Auto-generates responses every 5 seconds

## Tech Stack

- **MediaPipe**: Face, pose, hand detection
- **OpenCV**: Video processing
- **NumPy**: Data analysis
- **Gemini Vision API**: Board content analysis (optional)

## Detection Modes

### With MediaPipe

- 468 facial landmarks
- 33 body keypoints
- 21 hand keypoints per hand
- High precision

### Without MediaPipe (Fallback)

- Basic face detection only
- No pose/hand detection
- Limited functionality
