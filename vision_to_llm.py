import cv2
import easyocr
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# capture image from camera or use existing image
def capture_image(filename="board.png"):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")

        # check if an image file already exists
        if os.path.exists(filename):
            print(f"Found existing image: {filename}")
            use_existing = input(f"Use existing {filename}? (y/n): ").lower().strip()
            if use_existing == "y":
                return True

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
        # read a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video frame.")
            break

        # display the live video stream in a window
        cv2.imshow('Camera - Press "s" to save, "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(filename, frame)
            print(f"Success. Photo saved as {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return True
        elif key == ord("q"):
            print("Exiting program.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return False


# Run OCR
def extract_text_from_image(filename="board.png"):
    print("Initializing OCR Reader")
    reader = easyocr.Reader(["en", "ch_sim"])

    print(f"Reading text from {filename}")
    results = reader.readtext(filename)

    if not results:
        print("OCR failed to recognize any text.")
        return None

    board_text = " ".join([res[1] for res in results])
    print("---------------------------")
    print("OCR Results:")
    print(board_text)
    print("---------------------------")
    return board_text


# Send extracted text to llm API
def query_llm(text):
    if not text:
        return "Error: No text to send to llm."
    try:
        # Get API key from environment variable
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "Error: Please set DEEPSEEK_API_KEY environment variable"
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        print("Sending request to llm...")

        # modify the content here to change the instruction given to the llm
        prompt_content = f"Please answer the following question or summarize: '{text}'"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt_content}],
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return f"Error occurred while calling llm API: {e}"


if __name__ == "__main__":
    IMAGE_FILE = "board.png"

    result = capture_image(IMAGE_FILE)
    if result:
        image_to_use = result if isinstance(result, str) else IMAGE_FILE
        extracted_text = extract_text_from_image(image_to_use)
        llm_answer = query_llm(extracted_text)
        print("---------------------------")
        print("LLM Response:")
        print(llm_answer)
        print("---------------------------")
