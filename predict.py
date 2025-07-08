#!/usr/bin/env python3
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from sense_hat import SenseHat

# Config
MODEL_PATH = "cnn_gesture_model.tflite"
IMG_SIZE = (160, 160)  # Must match training input size
THRESH = 0.85          # Confidence threshold

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

# Initialize SenseHat and Picamera2
sense = SenseHat()
sense.clear()

picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (1640, 1232), "format": "RGB888"},  # same as your capture script
    lores={"size": (640, 480), "format": "RGB888"},
    display="lores"
)
picam2.configure(config)
picam2.start()

print("Starting prediction... Press 'q' to quit.")

try:
    while True:
        # Capture high-res frame to match training data
        frame = picam2.capture_array("main")  # 1640x1232 RGB

        # Resize to model input size
        resized = cv2.resize(frame, IMG_SIZE)

        # Normalize and expand dims
        input_tensor = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        prob = interpreter.get_tensor(output_index)[0][0]

        # LED color logic
        if prob > THRESH:
            sense.clear((0, 255, 0))  # Green = thumbs_up
            label = "Thumbs Up"
        else:
            sense.clear((255, 0, 0))  # Red = no thumbs_up
            label = "No Thumbs Up"

        # Show preview with label
        display_frame = cv2.resize(frame, (320, 240))
        cv2.putText(display_frame, f"{label}: {prob:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Gesture Prediction", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    sense.clear()
    picam2.stop()
    cv2.destroyAllWindows()
    print("Prediction stopped.")
