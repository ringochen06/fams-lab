FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.3-py3

RUN apt-get update && apt-get install -y python3-opencv libgl1-mesa-glx

WORKDIR /app
COPY . /app

RUN pip install easyocr openai

ENV DEEPSEEK_API_KEY=""
CMD ["python3", "ocr_llm_jetson.py"]
