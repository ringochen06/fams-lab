#!/bin/bash

# Check if this is a Jetson device
if [ "$(uname -m)" != "aarch64" ]; then
    echo "Not a Jetson device"
    exit 1
fi

echo "Setting up Jetson..."

# Update and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev python3-opencv libopencv-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran

# Create environment and install packages
python3 -m venv jetson_env
source jetson_env/bin/activate
pip install --upgrade pip

# Install Jetson-compatible packages
pip install opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.0.0 easyocr==1.7.2 openai==2.1.0 python-dotenv==1.1.1 scipy==1.16.2 scikit-image==0.25.2 python-bidi==0.6.6 PyYAML==6.0.3 Shapely==2.1.2 pyclipper==1.3.0.post6 ninja==1.13.0 httpx==0.28.1 anyio==4.11.0 distro==1.9.0 jiter==0.11.0 pydantic==2.11.9 sniffio==1.3.1 tqdm==4.67.1 typing-extensions==4.15.0

echo "Setup complete!"
echo "Usage: source jetson_env/bin/activate && python vision_to_llm_jetson.py"
