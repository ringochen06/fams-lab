#!/bin/bash
echo "Starting Jetson OCR + LLM container..."

sudo docker run -it --rm \
  --runtime nvidia --gpus all \
  --network host \
  --device /dev/video0 \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v $(pwd):/app \
  -e DEEPSEEK_API_KEY="your_real_api_key_here" \
  jetson-ocr-llm
