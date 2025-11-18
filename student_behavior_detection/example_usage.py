"""
Example usage of Student Behavior Detection
Demonstrates how to use the behavior detector in your own code
"""

import cv2
from behavior_detector import BehaviorDetector, StudentState


def example_basic_usage():
    """Basic example: detect behavior from camera"""
    print("Example 1: Basic Behavior Detection")
    print("-" * 50)

    detector = BehaviorDetector()
    cap = cv2.VideoCapture(0)

    frame_count = 0
    while frame_count < 100:  # Process 100 frames
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)

        # Check for specific behaviors
        if StudentState.NODDING_TOO_MUCH in results["states"]:
            print(f"Frame {frame_count}: Student nodding too much - skip details")

        if StudentState.NOT_UNDERSTANDING in results["states"]:
            print(f"Frame {frame_count}: Student not understanding - explain better")

        frame_count += 1

    cap.release()
    print("Done!")


def example_custom_response():
    """Example: Custom tutor responses based on behavior"""
    print("\nExample 2: Custom Tutor Responses")
    print("-" * 50)

    detector = BehaviorDetector()
    cap = cv2.VideoCapture(0)

    def get_tutor_action(states):
        """Generate tutor action based on student states"""
        if StudentState.NODDING_TOO_MUCH in states:
            return {
                "action": "skip_details",
                "message": "You seem to understand well. Let's move faster.",
            }
        elif StudentState.NOT_UNDERSTANDING in states:
            return {
                "action": "explain_differently",
                "message": "This seems unclear. Let me explain it another way.",
            }
        elif StudentState.TAKING_NOTES in states:
            return {
                "action": "continue",
                "message": "Good note-taking! Let's continue.",
            }
        elif StudentState.BORED in states:
            return {
                "action": "engage",
                "message": "Let's try a more engaging approach.",
            }
        else:
            return {"action": "normal", "message": "Continuing with explanation."}

    print("Starting detection... Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        action = get_tutor_action(results["states"])

        # Display on frame
        cv2.putText(
            frame,
            action["message"],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Custom Tutor Response", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def example_metrics_tracking():
    """Example: Track behavior metrics over time"""
    print("\nExample 3: Metrics Tracking")
    print("-" * 50)

    detector = BehaviorDetector()
    cap = cv2.VideoCapture(0)

    metrics_history = {
        "nod_frequency": [],
        "staring_episodes": 0,
        "note_taking_time": 0,
    }

    frame_count = 0
    while frame_count < 300:  # 10 seconds at 30 FPS
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        metrics = results["metrics"]

        # Track metrics
        if metrics["nod_frequency"] > 0:
            metrics_history["nod_frequency"].append(metrics["nod_frequency"])

        if metrics["is_staring"]:
            metrics_history["staring_episodes"] += 1

        if metrics["is_taking_notes"]:
            metrics_history["note_taking_time"] += 1

        frame_count += 1

    cap.release()

    # Print summary
    print("\nBehavior Summary:")
    if metrics_history["nod_frequency"]:
        avg_nod = sum(metrics_history["nod_frequency"]) / len(
            metrics_history["nod_frequency"]
        )
        print(f"  Average nod frequency: {avg_nod:.2f} nods/sec")
    print(f"  Staring episodes: {metrics_history['staring_episodes']}")
    print(f"  Note-taking time: {metrics_history['note_taking_time']} frames")


if __name__ == "__main__":
    print("Student Behavior Detection - Examples")
    print("=" * 50)

    # Uncomment the example you want to run:
    # example_basic_usage()
    # example_custom_response()
    example_metrics_tracking()
