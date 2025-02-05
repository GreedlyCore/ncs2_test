from ultralytics import YOLO

model = YOLO("/home/sonieth/ncs2_workspace/ncs2_test/workspace/models/yolov5m.pt")
# Export the model to ONNX format
model.export(format="onnx")  