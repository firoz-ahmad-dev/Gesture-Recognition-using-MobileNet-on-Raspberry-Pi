README.txt
==========

Project Title:
---------------
Gesture Recognition using MobileNet on Raspberry Pi


Description:
-------------
This project implements a lightweight, real-time gesture recognition system using MobileNetV2 and TensorFlow Lite on a Raspberry Pi. It allows users to collect gesture images, train a CNN model via transfer learning, and deploy it for real-time inference with LED feedback using the Sense HAT.

The primary gesture classification task distinguishes between “thumbs up” and “no thumbs up” gestures.


Repository Contents:
---------------------
- capture.py           : Script to collect gesture images using the Raspberry Pi camera
- predict.py           : Real-time inference script using the TFLite model and Sense HAT for LED feedback
- cnn_gesture_model.tflite : Trained MobileNetV2 model converted to TensorFlow Lite format
- gesture_data/        : Folder structure for captured images (not included in repo; to be created during data collection)
- Lab_Report.tex       : Final lab report in LaTeX format (optional)
- README.txt           : This file


Requirements:
--------------
- Raspberry Pi 4 (or similar)
- Pi Camera Module
- Sense HAT
- Python 3.x
- TensorFlow 2.x
- picamera2
- Pillow (PIL)
- NumPy


Setup Instructions:
--------------------
1. Clone this repository on your Raspberry Pi.
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2 sense-hat
   pip install tensorflow pillow numpy