import openvino as ov

ONNX_NLP_MODEL_PATH = "/workspace/models/yolov3-tinyu.onnx"
MODEL_DIRECTORY_PATH = "/workspace/models/"
ov_model = ov.convert_model(ONNX_NLP_MODEL_PATH)
# then model can be serialized to *.xml & *.bin files
ov.save_model(ov_model, MODEL_DIRECTORY_PATH / "distilbert3.xml")