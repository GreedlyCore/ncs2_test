print("Convert ONNX model to OpenVINO IR:")
onnx_path = "../workspace/models/yolov5m.onnx"
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
model_output_path = "../workspace/models"

# Construct the command for Model Optimizer.
command_mo = f"""mo
                 --input_model "{onnx_path}"
                 --input_shape "[1, 3, {IMAGE_HEIGHT}, {IMAGE_WIDTH}]"
                 --data_type FP32
                 --output_dir "{model_output_path}"
                 """
command_mo = " ".join(command_mo.split())
print(command_mo)