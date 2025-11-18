"""
Integrated AI Tutor System
Combines board content analysis (vision_api) with student behavior detection
"""

import cv2
import sys
import os
import time
from pathlib import Path

# Add parent directory to path to import vision_api
sys.path.append(str(Path(__file__).parent.parent))

from behavior_detector import BehaviorDetector, StudentState

# Try to import vision_api functions
try:
    from vision_api.vision_llm import query_vision_llm, capture_image

    VISION_API_AVAILABLE = True
except ImportError:
    print("Warning: vision_api not available. Board analysis disabled.")
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
    ) -> str:
        """Generate personalized tutor response based on behavior and content"""
        states = behavior_results.get("states", [])
        recommendations = behavior_results.get("recommendations", [])

        response_parts = []

        # Handle different student states
        if StudentState.NODDING_TOO_MUCH in states:
            response_parts.append(
                "I notice you're nodding a lot - that's great! It seems you understand this concept well."
            )
            response_parts.append(
                "Let's move on to the next topic to keep things interesting."
            )

        elif StudentState.NOT_UNDERSTANDING in states:
            response_parts.append(
                "I see you're looking at me but not taking notes. It seems this concept might be unclear."
            )
            response_parts.append("Let me explain it differently:")
            if board_content:
                response_parts.append(
                    f"Looking at what we have: {board_content[:200]}..."
                )
            response_parts.append(
                "Would you like me to break this down into smaller steps?"
            )

        elif StudentState.TAKING_NOTES in states:
            response_parts.append(
                "Great! I see you're taking notes. That's an excellent way to learn."
            )
            response_parts.append("Let me continue explaining at this pace.")

        elif StudentState.BORED in states:
            response_parts.append(
                "You seem a bit disengaged. Let me try a different approach to make this more interesting."
            )
            response_parts.append(
                "Would you like me to use an example or a different explanation method?"
            )

        elif StudentState.ENGAGED in states:
            response_parts.append(
                "You look engaged! Let's continue with the current explanation."
            )

        elif StudentState.DISTRACTED in states:
            response_parts.append(
                "I notice you might be distracted. Let's refocus on the topic."
            )

        else:
            response_parts.append(
                "I'm here to help. Let me know if you have any questions."
            )

        return " ".join(response_parts)

    def run_interactive_session(self):
        """Run interactive tutoring session"""
        print("=" * 60)
        print("AI Tutor System - Integrated Mode")
        print("=" * 60)
        print("\nThis system combines:")
        print("  1. Board content analysis (vision_api)")
        print("  2. Student behavior detection")
        print("\nPress 'b' to capture and analyze board")
        print("Press 'q' to quit")
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
            status_text = "Board: " + ("Analyzed" if board_analyzed else "Not analyzed")
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
                # Capture and analyze board
                print("\nCapturing board image...")
                if VISION_API_AVAILABLE:
                    image_path = capture_image("board.png")
                    if image_path:
                        print("Analyzing board content...")
                        board_content = self.analyze_board(image_path)
                        print(f"\nBoard Analysis:\n{board_content}\n")
                        board_analyzed = True

                        # Generate response based on behavior + board
                        response = self.generate_tutor_response(
                            behavior_results, board_content
                        )
                        print(f"\nTutor Response:\n{response}\n")
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

                if board_analyzed:
                    response = self.generate_tutor_response(
                        behavior_results, self.last_board_analysis
                    )
                    print(f"\nTutor Response:\n{response}")
                print("=" * 60)
                last_response_time = current_time

        cap.release()
        cv2.destroyAllWindows()


def main():
    tutor = IntegratedAITutor()
    tutor.run_interactive_session()


if __name__ == "__main__":
    main()
