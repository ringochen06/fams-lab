"""
Demo program - runs for 10 seconds to demonstrate behavior detection
"""

import cv2
import time
from behavior_detector import BehaviorDetector, StudentState


def main():
    print("=" * 70)
    print("Student Behavior Detection System - Demo Mode")
    print("=" * 70)
    print("\nSystem will run for 10 seconds to demonstrate real-time detection")
    print("Please ensure camera is connected and facing you")
    print("\nDetected behaviors include:")
    print("  - Nodding frequency (too much means understanding, can speed up)")
    print("  - Staring state (not taking notes may indicate confusion)")
    print("  - Note-taking actions")
    print("  - Body posture (engaged/bored)")
    print("\n" + "=" * 70)
    print("Starting detection... (will auto-stop after 10 seconds)")
    print("=" * 70 + "\n")

    detector = BehaviorDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. No other program is using the camera")
        print("  3. Camera permissions are granted")
        return

    start_time = time.time()
    frame_count = 0
    last_print_time = 0

    print("Detecting... (output results every 2 seconds)\n")

    try:
        while time.time() - start_time < 10:  # Run for 10 seconds
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Cannot read video frame")
                break

            frame_count += 1

            # Process frame
            results = detector.process_frame(frame)

            # Output results every 2 seconds
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                elapsed = int(current_time - start_time)
                print(f"\n[{elapsed}s] Detection Results:")
                print(f"  Face detected: {'✓' if results['face_detected'] else '✗'}")
                print(f"  Pose detected: {'✓' if results['pose_detected'] else '✗'}")
                print(f"  Hands detected: {'✓' if results['hands_detected'] else '✗'}")

                metrics = results["metrics"]
                print(f"  Nod frequency: {metrics['nod_frequency']:.2f} nods/sec")
                print(f"  Is staring: {'Yes' if metrics['is_staring'] else 'No'}")
                print(
                    f"  Taking notes: {'Yes' if metrics['is_taking_notes'] else 'No'}"
                )
                print(f"  Body posture: {metrics['posture']}")

                if results["states"]:
                    print(f"  Detected states: {[s.value for s in results['states']]}")

                if results["recommendations"]:
                    print(f"  Teaching recommendations:")
                    for rec in results["recommendations"][:2]:  # Show first 2 only
                        print(f"    • {rec}")

                last_print_time = current_time

            # Display on window (if possible)
            frame = detector.draw_detections(frame, results)
            cv2.imshow("Student Behavior Detection - Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser pressed 'q' to quit")
                break

        print("\n" + "=" * 70)
        print("Demo completed!")
        print("=" * 70)
        print(f"\nStatistics:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Average FPS: {frame_count / 10:.1f} FPS")
        print(
            f"  Detector mode: {'MediaPipe mode' if detector.mediapipe_available else 'OpenCV basic mode'}"
        )
        print("\n" + "=" * 70)
        print("Tip: Run full program with: python behavior_detector.py")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
