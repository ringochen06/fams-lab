"""

Student Behavior Detection for AI Tutor
Detects facial expressions and body language to personalize teaching approach.
"""

import cv2
import numpy as np
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Tuple, Optional


class StudentState(Enum):
    """Student engagement states"""

    ENGAGED = "engaged"
    CONFUSED = "confused"
    BORED = "bored"
    NODDING_TOO_MUCH = "nodding_too_much"
    NOT_UNDERSTANDING = "not_understanding"
    TAKING_NOTES = "taking_notes"
    DISTRACTED = "distracted"


class BehaviorDetector:
    """Detects and interprets student behavior from video feed"""

    def __init__(self, window_size: int = 30):
        """
        Initialize behavior detector.

        Args:
            window_size: Number of frames to analyze for behavior patterns
        """
        self.window_size = window_size

        # Initialize MediaPipe
        try:
            import mediapipe as mp

            self.mp = mp
            self.mp_face = mp.solutions.face_mesh
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands

            self.face_mesh = self.mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.mediapipe_available = True
        except ImportError:
            print(
                "Warning: MediaPipe not available. Using basic OpenCV face detection."
            )
            self.mediapipe_available = False
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # Tracking buffers
        self.head_positions = deque(maxlen=window_size)
        self.eye_states = deque(maxlen=window_size)
        self.hand_positions = deque(maxlen=window_size)
        self.pose_angles = deque(maxlen=window_size)

        # Thresholds
        self.NOD_THRESHOLD = 0.08  # Head movement threshold for nodding (increased to reduce false positives)
        self.NOD_FREQUENCY_THRESHOLD = (
            2.0  # Nods per second (increased to 2.0 - requires very frequent nodding)
        )
        self.MIN_NOD_AMPLITUDE = (
            8.0  # Minimum vertical movement in pixels for a valid nod
        )
        self.STARE_DURATION = 5.0  # Seconds of staring without movement
        self.EYE_CLOSED_RATIO = 0.3  # Eye aspect ratio for closed eyes

        # State tracking
        self.last_nod_time = 0
        self.nod_count = 0
        self.stare_start_time = None
        self.last_hand_movement = time.time()

    def detect_face_mediapipe(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect face using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Extract key points
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x * w, landmark.y * h])

        landmarks = np.array(landmarks)

        # Get head position (nose tip)
        nose_tip = landmarks[1]  # Nose tip landmark

        # Get eye landmarks for blink detection
        # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        left_eye = landmarks[
            [
                33,
                7,
                163,
                144,
                145,
                153,
                154,
                155,
                133,
                173,
                157,
                158,
                159,
                160,
                161,
                246,
            ]
        ]
        right_eye = landmarks[
            [
                362,
                382,
                381,
                380,
                374,
                373,
                390,
                249,
                263,
                466,
                388,
                387,
                386,
                385,
                384,
                398,
            ]
        ]

        # Calculate eye aspect ratio
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        return {
            "nose_tip": nose_tip,
            "eye_aspect_ratio": avg_ear,
            "landmarks": landmarks,
            "face_detected": True,
        }

    def detect_face_opencv(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect face using OpenCV Haar Cascade (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        # Approximate nose tip as center of face
        nose_tip = np.array([x + w / 2, y + h / 2])

        return {"nose_tip": nose_tip, "face_bbox": (x, y, w, h), "face_detected": True}

    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect body pose using MediaPipe"""
        if not self.mediapipe_available:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x * w, landmark.y * h, landmark.visibility])

        landmarks = np.array(landmarks)

        # Calculate body posture angle (shoulder to hip)
        # Pose landmarks: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        if (
            landmarks[11][2] > 0.5
            and landmarks[12][2] > 0.5
            and landmarks[23][2] > 0.5
            and landmarks[24][2] > 0.5
        ):
            shoulder_center = (landmarks[11][:2] + landmarks[12][:2]) / 2
            hip_center = (landmarks[23][:2] + landmarks[24][:2]) / 2

            # Calculate angle from vertical
            vec = shoulder_center - hip_center
            angle = np.arctan2(vec[0], vec[1]) * 180 / np.pi

            return {
                "landmarks": landmarks,
                "posture_angle": angle,
                "shoulder_center": shoulder_center,
                "hip_center": hip_center,
            }

        return {"landmarks": landmarks}

    def detect_hands(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect hand positions and movements"""
        if not self.mediapipe_available:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        h, w = frame.shape[:2]
        hand_positions = []

        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist position
            wrist = hand_landmarks.landmark[0]
            hand_positions.append([wrist.x * w, wrist.y * h])

        return {
            "hand_positions": np.array(hand_positions),
            "num_hands": len(hand_positions),
        }

    def analyze_nodding(self) -> Tuple[bool, float]:
        """Analyze if student is nodding too much - with stricter detection"""
        if len(self.head_positions) < 20:  # Need more frames for reliable detection
            return False, 0.0

        positions = np.array(list(self.head_positions))
        y_positions = positions[:, 1]

        # Filter out noise: calculate overall variance first
        y_variance = np.var(y_positions)
        if y_variance < 20:  # Too stable, no significant movement
            return False, 0.0

        # Detect actual nodding patterns: must have clear up-down-up cycles
        # with sufficient amplitude
        movement_threshold = self.NOD_THRESHOLD * 100  # Convert to pixels
        min_amplitude = self.MIN_NOD_AMPLITUDE

        # Track peaks and valleys to identify complete nod cycles
        peaks = []  # Local maxima (head up)
        valleys = []  # Local minima (head down)

        # Find local extrema with sufficient amplitude
        for i in range(2, len(y_positions) - 2):
            # Check for local maximum (peak)
            if (
                y_positions[i] > y_positions[i - 1]
                and y_positions[i] > y_positions[i + 1]
                and y_positions[i] > y_positions[i - 2]
                and y_positions[i] > y_positions[i + 2]
            ):
                peaks.append((i, y_positions[i]))

            # Check for local minimum (valley)
            if (
                y_positions[i] < y_positions[i - 1]
                and y_positions[i] < y_positions[i + 1]
                and y_positions[i] < y_positions[i - 2]
                and y_positions[i] < y_positions[i + 2]
            ):
                valleys.append((i, y_positions[i]))

        # Count valid nod cycles: must have valley->peak->valley pattern
        # with sufficient amplitude between peak and valley
        nod_cycles = 0
        if len(peaks) > 0 and len(valleys) > 0:
            # Combine and sort by position
            all_extrema = sorted(peaks + valleys, key=lambda x: x[0])

            # Look for valley-peak-valley patterns (complete nod cycle)
            for i in range(len(all_extrema) - 2):
                p1, y1 = all_extrema[i]
                p2, y2 = all_extrema[i + 1]
                p3, y3 = all_extrema[i + 2]

                # Check if pattern is valley-peak-valley or peak-valley-peak
                # (both represent a nod cycle)
                if (y1 < y2 > y3) or (y1 > y2 < y3):
                    # Calculate amplitude
                    amplitude = abs(max(y1, y2, y3) - min(y1, y2, y3))
                    if amplitude >= min_amplitude:
                        nod_cycles += 1

        # Calculate frequency based on actual nod cycles
        if len(positions) > 0:
            time_window = len(positions) / 30.0  # seconds (assuming ~30 FPS)
            if time_window > 0:
                frequency = nod_cycles / time_window
            else:
                frequency = 0.0
        else:
            frequency = 0.0

        # Very strict conditions: high frequency AND multiple cycles AND sufficient variance
        is_nodding_too_much = (
            frequency > self.NOD_FREQUENCY_THRESHOLD
            and nod_cycles >= 3  # Need at least 3 complete cycles
            and y_variance > 30  # Overall movement must be significant
        )

        return is_nodding_too_much, frequency

    def analyze_staring(self) -> bool:
        """Analyze if student is staring without movement"""
        if len(self.head_positions) < 30:
            return False

        positions = np.array(list(self.head_positions))

        # Check if head position is very stable (low variance)
        head_variance = np.var(positions, axis=0)
        is_stable = np.all(head_variance < 50)  # Low movement threshold

        # Check if eyes are open (not sleeping)
        if len(self.eye_states) > 0:
            avg_ear = np.mean(list(self.eye_states))
            eyes_open = avg_ear > self.EYE_CLOSED_RATIO
        else:
            eyes_open = True

        # Check if hands are not moving (not taking notes)
        hands_moving = False
        if len(self.hand_positions) > 10:
            hand_pos = np.array(list(self.hand_positions))
            hand_variance = np.var(hand_pos, axis=0)
            hands_moving = np.any(hand_variance > 100)

        # Staring = stable head + eyes open + hands not moving
        is_staring = is_stable and eyes_open and not hands_moving

        return is_staring

    def analyze_note_taking(self) -> bool:
        """Analyze if student is taking notes"""
        if len(self.hand_positions) < 10:
            return False

        hand_pos = np.array(list(self.hand_positions))

        # Check hand movement variance
        if len(hand_pos) > 0:
            hand_variance = np.var(hand_pos, axis=0)
            # Significant hand movement suggests note-taking
            is_taking_notes = np.any(hand_variance > 200)
            return is_taking_notes

        return False

    def analyze_posture(self) -> str:
        """Analyze body posture"""
        if len(self.pose_angles) < 10:
            return "unknown"

        angles = list(self.pose_angles)
        avg_angle = np.mean([a for a in angles if a is not None])

        if avg_angle is None:
            return "unknown"

        # Negative angle = leaning forward (engaged)
        # Positive angle = leaning back (bored)
        if avg_angle < -10:
            return "engaged"
        elif avg_angle > 10:
            return "bored"
        else:
            return "neutral"

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame and update tracking"""
        current_time = time.time()

        # Detect face
        if self.mediapipe_available:
            face_data = self.detect_face_mediapipe(frame)
        else:
            face_data = self.detect_face_opencv(frame)

        # Detect pose
        pose_data = self.detect_pose(frame)

        # Detect hands
        hand_data = self.detect_hands(frame)

        # Update tracking buffers
        if face_data and "nose_tip" in face_data:
            self.head_positions.append(face_data["nose_tip"])
            if "eye_aspect_ratio" in face_data:
                self.eye_states.append(face_data["eye_aspect_ratio"])

        if pose_data and "posture_angle" in pose_data:
            self.pose_angles.append(pose_data["posture_angle"])

        if hand_data and "hand_positions" in hand_data:
            if len(hand_data["hand_positions"]) > 0:
                # Use first hand position
                self.hand_positions.append(hand_data["hand_positions"][0])
                self.last_hand_movement = current_time

        # Analyze behaviors
        is_nodding_too_much, nod_frequency = self.analyze_nodding()
        is_staring = self.analyze_staring()
        is_taking_notes = self.analyze_note_taking()
        posture = self.analyze_posture()

        # Determine overall state
        states = []
        recommendations = []

        if is_nodding_too_much:
            states.append(StudentState.NODDING_TOO_MUCH)
            recommendations.append(
                "Student is nodding too much - skip detailed explanations, move faster"
            )

        if is_staring and not is_taking_notes:
            states.append(StudentState.NOT_UNDERSTANDING)
            recommendations.append(
                "Student is staring without taking notes - explain concept more clearly and check understanding"
            )

        if is_taking_notes:
            states.append(StudentState.TAKING_NOTES)
            recommendations.append(
                "Student is taking notes - good engagement, continue at current pace"
            )

        if posture == "bored":
            states.append(StudentState.BORED)
            recommendations.append(
                "Student appears bored - try to make content more engaging"
            )
        elif posture == "engaged":
            states.append(StudentState.ENGAGED)

        if not face_data:
            states.append(StudentState.DISTRACTED)
            recommendations.append(
                "Student not detected - may be distracted or out of frame"
            )

        return {
            "face_detected": face_data is not None,
            "pose_detected": pose_data is not None,
            "hands_detected": hand_data is not None,
            "states": states,
            "recommendations": recommendations,
            "metrics": {
                "nod_frequency": nod_frequency,
                "is_staring": is_staring,
                "is_taking_notes": is_taking_notes,
                "posture": posture,
            },
            "face_data": face_data,
            "pose_data": pose_data,
            "hand_data": hand_data,
        }

    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        # Draw face landmarks
        if results["face_data"] and self.mediapipe_available:
            face_landmarks = results["face_data"].get("landmarks")
            if face_landmarks is not None:
                for point in face_landmarks[::10]:  # Draw every 10th point
                    cv2.circle(
                        frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1
                    )

        # Draw pose
        if results["pose_data"] and "landmarks" in results["pose_data"]:
            landmarks = results["pose_data"]["landmarks"]
            # Draw key points
            key_points = [11, 12, 23, 24]  # Shoulders and hips
            for idx in key_points:
                if landmarks[idx][2] > 0.5:
                    cv2.circle(
                        frame,
                        (int(landmarks[idx][0]), int(landmarks[idx][1])),
                        5,
                        (255, 0, 0),
                        -1,
                    )

        # Draw hands
        if results["hand_data"] and "hand_positions" in results["hand_data"]:
            for pos in results["hand_data"]["hand_positions"]:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, (0, 0, 255), -1)

        # Draw text overlay
        y_offset = 30
        cv2.putText(
            frame,
            f"States: {[s.value for s in results['states']]}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        for i, rec in enumerate(
            results["recommendations"][:3]
        ):  # Show max 3 recommendations
            y_offset += 25
            cv2.putText(
                frame,
                rec[:60],
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        return frame


def main():
    """Main function to run behavior detection"""
    detector = BehaviorDetector()

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Starting behavior detection...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = detector.process_frame(frame)

        # Draw detections
        frame = detector.draw_detections(frame, results)

        # Display
        cv2.imshow("Student Behavior Detection", frame)

        # Print recommendations to console
        if results["recommendations"]:
            print("\n" + "=" * 60)
            print("Recommendations:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")
            print("=" * 60)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
