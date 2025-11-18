import cv2
import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()


def capture_image(filename="board.png", use_gstreamer=True):
    """Capture one image from Jetson camera or USB camera."""
    if use_gstreamer:
        gst_str = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width=640,height=480,format=NV12,framerate=30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw,format=BGRx ! "
            "videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None

    print("Capturing photo in 3 seconds...")
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read frame.")
        return None

    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")
    return filename


def encode_image(image_path):
    """Read image data for Gemini API."""
    with open(image_path, "rb") as f:
        return f.read()


def query_vision_llm(image_path):
    """
    Send image directly to Gemini Vision API.

    Args:
        image_path: Path to the image file
    """
    if not os.path.exists(image_path):
        return "Error: Image not found."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Missing GEMINI_API_KEY environment variable.\nYou can set it with: export GEMINI_API_KEY='your_key'"

    try:
        # Initialize Gemini Client
        client = genai.Client(api_key=api_key)

        # Read image data
        image_data = encode_image(image_path)

        print("Sending image to Gemini Vision API...")

        prompt = (
            "Please analyze this whiteboard image. If it contains text, mathematical equations, "
            "diagrams, or charts, please describe and explain them in detail. "
            "For equations, provide their meaning and possible solutions. "
            "For diagrams or charts, describe their structure and meaning."
        )

        # Generate response with image using new API
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Fast and free
            # Alternative: model="gemini-2.5-pro"  # Better quality
            contents=[
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": image_data}},
                        {"text": prompt},
                    ]
                }
            ],
        )

        return response.text

    except Exception as e:
        return f"Error occurred while calling Gemini Vision API: {e}"


if __name__ == "__main__":

    image_path = capture_image("board.png")

    if image_path:
        # Use Gemini Vision API
        print("\nAttempting to use Gemini Vision API...")
        result = query_vision_llm(image_path)

        if "Error" in result:
            print("\n" + "=" * 60)
            print("⚠️  API Key not set or incorrect")
            print("=" * 60)
            print("\nPlease set GEMINI_API_KEY:")
            print("  export GEMINI_API_KEY='your_api_key'")
            print("\nOr create a .env file:")
            print("  GEMINI_API_KEY=your_api_key")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("Gemini Vision Response:")
        print("=" * 60)
        print(result)
        print("=" * 60)
    else:
        print("Capture failed.")
