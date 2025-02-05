import torch
import onnx
import subprocess
from pathlib import Path
from ultralytics import YOLO
from openvino.runtime import serialize, Core

# PyTorch --> ONNX --> OpenVINO
PTH_MODEL_PATH = "yolov5nu.pt"     
ONNX_MODEL_PATH = "yolov5nu.onnx"  
OPENVINO_MODEL_DIR = "openvino_model" 

# Step 1: Convert PyTorch (best.pt) to ONNX
def convert_pt_to_onnx():
    print("🔄 Converting PyTorch to ONNX...")
    
    # Load YOLOv5 model
    model = YOLO(PTH_MODEL_PATH)  

    # Export to ONNX format
    model.export(format="onnx", dynamic=True)  # Dynamic batch size
    print(f"✅ ONNX model saved at: {ONNX_MODEL_PATH}")

# Step 2: Convert ONNX to OpenVINO IR (.xml, .bin)
def convert_onnx_to_openvino():
    print("🔄 Converting ONNX to OpenVINO IR...")
    # Ensure OpenVINO Model Optimizer is installed
    mo_command = [
        "mo",
        "--input_model", ONNX_MODEL_PATH,
        "--output_dir", OPENVINO_MODEL_DIR,
        "--input_shape", "[1,3,640,640]",  # Standard YOLO input size
        "--data_type", "FP16"
    ]

    # Run Model Optimizer
    subprocess.run(mo_command, check=True)
    print(f"✅ OpenVINO model saved in: {OPENVINO_MODEL_DIR}")

if __name__ == "__main__":
    convert_pt_to_onnx()
    convert_onnx_to_openvino()
    print("🎯 Conversion complete: best.pt ➝ best.onnx ➝ best.xml & best.bin")
