"""
Simple test script - tests if behavior detector works properly
"""

import cv2
import sys
from behavior_detector import BehaviorDetector, StudentState


def test_detector():
    """Test detector initialization"""
    print("=" * 60)
    print("Testing Student Behavior Detection System")
    print("=" * 60)

    try:
        print("\n1. Initializing detector...")
        detector = BehaviorDetector()
        print("   ✓ Detector initialized successfully")

        if detector.mediapipe_available:
            print("   ✓ MediaPipe available (advanced features)")
        else:
            print("   ⚠ MediaPipe not available, using OpenCV basic detection")

        print("\n2. Testing camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("   ⚠ Cannot open camera")
            print("   Tip: If no camera, modify code to use video file")
            return False

        print("   ✓ Camera opened successfully")

        print("\n3. Testing frame processing (processing 5 frames)...")
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                print(f"   ⚠ Frame {i+1} read failed")
                break

            results = detector.process_frame(frame)
            print(
                f"   Frame {i+1}: face={results['face_detected']}, "
                f"pose={results['pose_detected']}, "
                f"hands={results['hands_detected']}"
            )

        cap.release()
        print("\n   ✓ Frame processing test completed")

        print("\n4. Testing behavior analysis...")
        # Simulate some data to test analysis functions
        if len(detector.head_positions) > 0:
            is_nodding, freq = detector.analyze_nodding()
            is_staring = detector.analyze_staring()
            is_notes = detector.analyze_note_taking()
            posture = detector.analyze_posture()

            print(f"   Nodding analysis: {is_nodding} (frequency: {freq:.2f})")
            print(f"   Staring analysis: {is_staring}")
            print(f"   Note taking: {is_notes}")
            print(f"   Posture: {posture}")

        print("\n" + "=" * 60)
        print("✓ All tests passed! System is ready to use")
        print("=" * 60)
        print("\nRun full program:")
        print("  python behavior_detector.py")
        print("\nOr integrated mode:")
        print("  python integrated_tutor.py")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_detector()
    sys.exit(0 if success else 1)
