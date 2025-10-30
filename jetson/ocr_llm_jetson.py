import cv2
import easyocr
import openai
import os
import time


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


def extract_text_from_image(filename="board.png"):
    """Extract text from image using EasyOCR."""
    if not os.path.exists(filename):
        print("Error: Image not found.")
        return None

    reader = easyocr.Reader(["en", "ch_sim"])
    results = reader.readtext(filename)

    if not results:
        print("OCR failed to detect text.")
        return None

    text = " ".join([res[1] for res in results])
    print("OCR Result:", text)
    return text


def query_llm(text):
    """Send OCR result to LLM API."""
    if not text:
        return "No text to send."

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: Missing DEEPSEEK_API_KEY environment variable."

    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        prompt = f"Summarize or answer based on this text: '{text}'"
        response = client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM API Error: {e}"


if __name__ == "__main__":
    print("Starting OCR + LLM pipeline...")
    image_path = capture_image("board.png")

    if image_path:
        extracted_text = extract_text_from_image(image_path)
        llm_output = query_llm(extracted_text)
        print("LLM Response:", llm_output)
    else:
        print("Capture failed.")
