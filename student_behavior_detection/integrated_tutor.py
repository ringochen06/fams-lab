"""
Integrated AI Tutor System
Combines board content analysis (vision_api) with student behavior detection
"""

import cv2
import sys
import os
import time
import threading
from pathlib import Path

# Add parent directory to path to import vision_api
# Get the absolute path of this file first
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

from behavior_detector import BehaviorDetector, StudentState

# Try to import vision_api functions
VISION_API_AVAILABLE = False
try:
    from vision_api.vision_llm import query_vision_llm, capture_image

    VISION_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: vision_api not available. Board analysis disabled.")
    print(f"  Import error: {e}")
    print(f"  Make sure vision_api directory exists at: {parent_dir / 'vision_api'}")
    print(f"  And that vision_llm.py contains the required functions.")
    VISION_API_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error loading vision_api: {e}")
    VISION_API_AVAILABLE = False


class IntegratedAITutor:
    """Combines board analysis with student behavior detection"""

    def __init__(self):
        self.behavior_detector = BehaviorDetector()
        self.current_topic = None
        self.last_board_analysis = None

    def analyze_board(self, image_path: str = "board.png") -> str:
        """Analyze board content using vision_api"""
        if not VISION_API_AVAILABLE:
            return "Board analysis not available"

        try:
            result = query_vision_llm(image_path)
            self.last_board_analysis = result
            return result
        except Exception as e:
            return f"Error analyzing board: {e}"

    def generate_tutor_response(
        self, behavior_results: dict, board_content: str = None
    ) -> dict:
        """
        Generate personalized tutor response based on behavior and content.
        Returns different structured values for each state.

        Returns:
            dict: Structured response containing:
                - state: Primary detected state
                - states: All detected states
                - message: Response message
                - action: Recommended action
                - priority: Priority level (1=critical, 2=important, 3=normal)
                - pace: Recommended teaching pace (fast/normal/slow)
                - needs_attention: Whether immediate attention is needed
                - metrics: Relevant metrics for this state
        """
        states = behavior_results.get("states", [])
        recommendations = behavior_results.get("recommendations", [])
        metrics = behavior_results.get("metrics", {})

        # Priority 1: Critical issues (distraction, not understanding)
        if StudentState.DISTRACTED in states:
            return {
                "state": StudentState.DISTRACTED,
                "states": states,
                "message": "I notice you might be distracted. Let's refocus on the topic.",
                "action": "refocus",
                "priority": 1,
                "pace": "pause",
                "needs_attention": True,
                "metrics": metrics,
                "response_type": "critical_issue",
            }

        if StudentState.NOT_UNDERSTANDING in states:
            message_parts = [
                "I see you're looking at me but not taking notes. It seems this concept might be unclear.",
                "Let me explain it differently:",
            ]
            if board_content:
                message_parts.append(
                    f"Looking at what we have: {board_content[:200]}..."
                )
            message_parts.append(
                "Would you like me to break this down into smaller steps?"
            )

            return {
                "state": StudentState.NOT_UNDERSTANDING,
                "states": states,
                "message": " ".join(message_parts),
                "action": "re_explain",
                "priority": 1,
                "pace": "slow",
                "needs_attention": True,
                "metrics": metrics,
                "response_type": "critical_issue",
                "board_content": board_content[:200] if board_content else None,
            }

        # Priority 2: Engagement indicators (can combine multiple positive behaviors)
        if StudentState.NODDING_TOO_MUCH in states:
            if StudentState.TAKING_NOTES in states:
                # Both nodding and taking notes - very engaged
                return {
                    "state": StudentState.NODDING_TOO_MUCH,
                    "states": states,
                    "message": "Excellent! I see you're nodding and taking notes - you're very engaged with this material. You seem to understand well. Let's continue at this pace.",
                    "action": "continue",
                    "priority": 2,
                    "pace": "normal",
                    "needs_attention": False,
                    "metrics": {
                        **metrics,
                        "engagement_level": "high",
                        "nod_frequency": metrics.get("nod_frequency", 0),
                    },
                    "response_type": "positive_engagement",
                    "combined_states": [
                        StudentState.NODDING_TOO_MUCH,
                        StudentState.TAKING_NOTES,
                    ],
                }
            else:
                # Just nodding
                return {
                    "state": StudentState.NODDING_TOO_MUCH,
                    "states": states,
                    "message": "I notice you're nodding a lot - that's great! It seems you understand this concept well. Let's move on to the next topic to keep things interesting.",
                    "action": "accelerate",
                    "priority": 2,
                    "pace": "fast",
                    "needs_attention": False,
                    "metrics": {
                        **metrics,
                        "nod_frequency": metrics.get("nod_frequency", 0),
                    },
                    "response_type": "understanding_confirmed",
                }

        if StudentState.TAKING_NOTES in states:
            # Just taking notes
            return {
                "state": StudentState.TAKING_NOTES,
                "states": states,
                "message": "Great! I see you're taking notes. That's an excellent way to learn. Let me continue explaining at this pace.",
                "action": "maintain_pace",
                "priority": 2,
                "pace": "normal",
                "needs_attention": False,
                "metrics": metrics,
                "response_type": "active_learning",
            }

        # Priority 3: Posture-based states
        if StudentState.BORED in states:
            return {
                "state": StudentState.BORED,
                "states": states,
                "message": "You seem a bit disengaged. Let me try a different approach to make this more interesting. Would you like me to use an example or a different explanation method?",
                "action": "change_approach",
                "priority": 3,
                "pace": "varied",
                "needs_attention": True,
                "metrics": {**metrics, "posture": metrics.get("posture", "bored")},
                "response_type": "engagement_issue",
            }

        if StudentState.ENGAGED in states:
            return {
                "state": StudentState.ENGAGED,
                "states": states,
                "message": "You look engaged! Let's continue with the current explanation.",
                "action": "continue",
                "priority": 3,
                "pace": "normal",
                "needs_attention": False,
                "metrics": {**metrics, "posture": metrics.get("posture", "engaged")},
                "response_type": "positive_engagement",
            }

        # Default response if no specific state detected
        return {
            "state": None,
            "states": states,
            "message": "I'm here to help. Let me know if you have any questions.",
            "action": "monitor",
            "priority": 3,
            "pace": "normal",
            "needs_attention": False,
            "metrics": metrics,
            "response_type": "neutral",
        }

    def _analyze_board_async(self, image_path):
        """Analyze board asynchronously"""
        try:
            board_content = self.analyze_board(image_path)

            if "Error" not in board_content:
                print(f"\n✅ Board Analysis:\n{board_content}\n")
                self.last_board_analysis = board_content
                print(
                    "Board analysis complete! You can now get personalized responses."
                )
            else:
                print(f"\n❌ {board_content}\n")
        except Exception as e:
            print(f"\n❌ Error analyzing board: {e}\n")

    def run_interactive_session(self):
        """Run interactive tutoring session"""
        print("=" * 60)
        print("AI Tutor System - Integrated Mode")
        print("=" * 60)
        print("\nThis system combines:")
        print("  1. Board content analysis (vision_api)")
        print("  2. Student behavior detection")
        print("\nControls:")
        print("  Press 'b' - Capture current frame and analyze board")
        print("  Press 'q' - Quit")
        print("=" * 60)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        board_analyzed = False
        last_response_time = 0
        response_interval = 5.0  # Print responses every 5 seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process behavior detection
            behavior_results = self.behavior_detector.process_frame(frame)

            # Draw detections
            frame = self.behavior_detector.draw_detections(frame, behavior_results)

            # Add status text
            board_status = "Analyzed" if self.last_board_analysis else "Not analyzed"
            status_text = f"Board: {board_status}"
            cv2.putText(
                frame,
                status_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Integrated AI Tutor", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):
                # Capture current frame and analyze board
                print("Capturing current frame as board image...")
                if VISION_API_AVAILABLE:
                    image_path = "board.png"
                    cv2.imwrite(image_path, frame)
                    print(f"✓ Image saved as {image_path}")

                    # Show loading message on frame
                    cv2.putText(
                        frame,
                        "Analyzing board content... Please wait...",
                        (10, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow("Integrated AI Tutor", frame)
                    cv2.waitKey(1)  # Update display

                    print(
                        "Sending to Gemini Vision API... (this may take a few seconds)"
                    )
                    # Analyze in background thread to avoid blocking
                    analysis_thread = threading.Thread(
                        target=self._analyze_board_async,
                        args=(image_path,),
                        daemon=True,
                    )
                    analysis_thread.start()
                else:
                    print("Board analysis not available (vision_api not found)")

            elif key == ord("q"):
                break

            # Auto-generate responses based on behavior (throttled)
            current_time = time.time()
            if (
                behavior_results["recommendations"]
                and current_time - last_response_time > response_interval
            ):
                print("\n" + "=" * 60)
                print("Current Behavior Analysis:")
                for rec in behavior_results["recommendations"]:
                    print(f"  - {rec}")

                if self.last_board_analysis:
                    response = self.generate_tutor_response(
                        behavior_results, self.last_board_analysis
                    )
                    # Display structured response
                    print(f"\nTutor Response:")
                    print(
                        f"  State: {response['state'].value if response['state'] else 'None'}"
                    )
                    print(
                        f"  Priority: {response['priority']} ({'Needs Attention' if response['needs_attention'] else 'Normal'})"
                    )
                    print(f"  Action: {response['action']}")
                    print(f"  Pace: {response['pace']}")
                    print(f"  Message: {response['message']}")
                    if "combined_states" in response:
                        print(
                            f"  Combined States: {[s.value for s in response['combined_states']]}"
                        )
                print("=" * 60)
                last_response_time = current_time

        cap.release()
        cv2.destroyAllWindows()


def main():
    tutor = IntegratedAITutor()
    tutor.run_interactive_session()


if __name__ == "__main__":
    main()
