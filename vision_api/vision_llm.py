import cv2
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()


def capture_image(filename="board.png"):
    """Capture image from camera or use existing image."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")

        # check if an image file already exists
        if os.path.exists(filename):
            print(f"Found existing image: {filename}")
            use_existing = input(f"Use existing {filename}? (y/n): ").lower().strip()
            if use_existing == "y":
                return filename

        # if no existing image detected, ask for an image file
        user_image = input(
            "Please provide the path to your image file to continue: "
        ).strip()

        if user_image and os.path.exists(user_image):
            print(f"Using image: {user_image}")
            return user_image
        else:
            print("No valid image file provided.")
            return False

    print("Camera is on. Press 's' to save photo, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video frame.")
            break

        cv2.imshow('Camera - Press "s" to save, "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(filename, frame)
            print(f"Success. Photo saved as {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return filename
        elif key == ord("q"):
            print("Exiting program.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return False


def encode_image(image_path):
    """Read image data for Gemini API."""
    with open(image_path, "rb") as f:
        return f.read()


def query_vision_llm(image_path):
    """
    Send image directly to Gemini Vision API.

    This replaces the OCR step - the Vision API can directly understand:
    - Text content
    - Mathematical equations
    - Diagrams and charts
    - Spatial relationships
    """
    if not os.path.exists(image_path):
        return "Error: Image not found."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Please set GEMINI_API_KEY environment variable\nYou can set it with: export GEMINI_API_KEY='your_key'"

    try:
        # Initialize Gemini Client
        client = genai.Client(api_key=api_key)

        # Read image data
        image_data = encode_image(image_path)

        print("Sending image to Gemini Vision API...")

        prompt = (
            "Please analyze this whiteboard image in detail. If the image contains:\n"
            "1. Text content - extract and summarize it\n"
            "2. Mathematical equations or formulas - identify the equations and explain their meaning, provide solutions if possible\n"
            "3. Diagrams, charts, or geometric shapes - describe their structure and meaning in detail\n"
            "4. Flowcharts or arrows - explain their logical relationships\n"
            "Please provide a comprehensive analysis."
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
    IMAGE_FILE = "board.png"

    print("=" * 60)
    print("Vision API - Using Gemini Vision API")
    print("=" * 60)
    print("\nFeatures:")
    print("✅ Fast response, no local model required")
    print("✅ No large memory or storage space needed")
    print("✅ Can understand mathematical equations and formulas")
    print("✅ Can describe diagrams, charts, and geometric shapes")
    print("=" * 60)
    print("\nNote: GEMINI_API_KEY environment variable is required\n")

    result = capture_image(IMAGE_FILE)
    if result:
        image_to_use = result if isinstance(result, str) else IMAGE_FILE

        # Use Gemini Vision API
        print("\nAttempting to use Gemini Vision API...")
        llm_answer = query_vision_llm(image_to_use)

        if "Error" in llm_answer:
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
        print(llm_answer)
        print("=" * 60)
