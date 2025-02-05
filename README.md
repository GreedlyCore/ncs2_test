# ncs2-test

| Details                 |               |
|-------------------------|---------------|
| Neural network:         |YOLOv8 or any in suitable format|
| Intel OpenVINO ToolKit: |OpenVINO 2022.1.0|
| Hardware Used:          | Linux laptop, Raspberri Pi 4 model b       |
| Device:                 | Neural Cumpute Stick 2 or CPU |


![Detected](https://github.com/rydikov/argus/raw/main/res/detected.jpg)

Not all OpenVINO versions suits for deploying ncs2 - please check
[URL1](https://github.com/openvinotoolkit/openvino/issues/14918)
[URL2](https://github.com/openvinotoolkit/openvino/releases/tag/2022.3.0).

Support of that device is now discontinued, this is why we should install any dependencies carefully and will be using docker + python venv environment.

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

## Architecture

The frames put to the queue from cameras in different threads.
In the main thread, the frames get from a queue and send to asynchronous recognition. Also in main thread the results are received and processed.

## Installation - locally

0. Git clone this project

1. Creating and entering venv && installing OpenVINO python wrapper via pip.
```bash
cd <INSTALL_DIR>/ncs2_workspace
python3 -m venv openvino_env
source openvino_env/bin/activate
pip install -r /ncs2_test/r.txt
```

2. Download docs and code examples for archived OpenVINO version, [here](https://docs.openvino.ai/archives/index.html)

3. Install OpenVINO Runtime and Dev Tools using apt or whatever, from this LOCAL site that you can grab from previous step.
```url
  openvino_docs_install_guides_installing_openvino_linux_header.html
```

4. Don't forget about optional, but necessary installations
```bash
sudo apt install openvino-opencv==2022.1.0
```
And setup for ncs2:
```url
openvino_docs_install_guides_configurations_for_ncs2.html
```
or
```bash
sudo usermod -a -G users "$(whoami)"
cd /opt/intel/openvino_2022.1.0.643/install_dependencies
bash sudo install_openvino_dependencies.sh
bash sudo install_NCS_udev_rules.sh
```

5. You can source for OpenVINO using such command, do it as much as possible or add to ~/bashrc
```bash
source /opt/intel/openvino_2022.1.0.643/setupvars.sh
```

6. Check available devices for deploy with this script inside workspace folder
```python
python3 show_devices.py
```
BTW, we pretty interested in MYRIAD device, so part of output should be like that:
```
        Device: MYRIAD
        Metrics:
                IMPORT_EXPORT_SUPPORT: True
                DEVICE_THERMAL: UNSUPPORTED TYPE
                DEVICE_ARCHITECTURE: MYRIAD
                OPTIMIZATION_CAPABILITIES: EXPORT_IMPORT, FP16
                RANGE_FOR_ASYNC_INFER_REQUESTS: 3, 6, 1
        <..AND..SO..ON....>
```

6. Test with demo deploy on myriad device (maybe source?)
```bash
cd ~/ncs2_workspace/ncs2_test/workspace/app
python3 classification_sample_async.py -i ../data/image/coco.jpg -m ../models/yolov3-tinyu.xml -d MYRIAD
```

7. Live inference with webcamera and pretrained YOLO weights ...
```
live_inference.py
```
omz_models_model_ssdlite_mobilenet_v2.html

## Installation - docker

Run application
```bash
docker-compose up
```

3. Build docker image
```bash
docker-compose build
```

Run application
```bash
docker-compose up
```


## Credit

- [Argus](https://github.com/rydikov/argus)
- [PINTO](https://github.com/PINTO0309/OpenVINO-YoloV3)
- [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
- [Yolov9](https://github.com/WongKinYiu/yolov9)
- [Aqara](https://developer.aqara.com/?lang=en)
