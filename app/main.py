import cv2
import numpy as np
from openvino.runtime import Core

# Load OpenVINO model
ie = Core()
model = ie.read_model(model="openvino_model/best.xml")
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")  # Run on Intel NCS2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# YOLOv5 input size
input_size = (640, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame for YOLOv5
    image_resized = cv2.resize(frame, input_size)
    image_resized = image_resized.transpose((2, 0, 1))  # Convert HWC to CHW
    image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32)  # Add batch dim

    # Run inference
    output = compiled_model([image_resized])

    # Dummy bounding box (replace with real YOLO processing)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = int(0.2 * w), int(0.2 * h), int(0.6 * w), int(0.6 * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Object", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("YOLOv5 on NCS2", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
