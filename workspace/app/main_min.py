
# # Hello Image Classification
# 
# This basic introduction to OpenVINO™ shows how to do inference with an image classification model.
# 
# A pre-trained [MobileNetV3 model](https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v3_small_1_0_224_tf.html) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) is used in this tutorial. For more information about how OpenVINO IR models are created, refer to the [TensorFlow to OpenVINO](../101-tensorflow-to-openvino/101-tensorflow-to-openvino.ipynb) tutorial.

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import time

# ## Load the Model


ie = Core()
# print(f"\n {ie.get_versions("MYRIAD")} \n")



ie.set_property("MYRIAD", {"MYRIAD_ENABLE_FORCE_RESET": "YES"})
ie.set_property("MYRIAD", {"MYRIAD_ENABLE_MX_BOOT": "NO"})
ie.set_property("MYRIAD", {"LOG_LEVEL": "LOG_DEBUG"})

time.sleep(2.0)

# model = ie.read_model(model="../models/v3-small_224_1.0_float.xml")
model = ie.read_model(model="../models/yolov3-tinyu.xml")
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")

output_layer = compiled_model.output(0)


# ## Load an Image


# The MobileNet model expects images in RGB format.
image = cv2.cvtColor(cv2.imread(filename="../data/image/coco.jpg"), code=cv2.COLOR_BGR2RGB)

# Resize to MobileNet image shape.
input_image = cv2.resize(src=image, dsize=(224, 224))

# Reshape to model input shape.
input_image = np.expand_dims(input_image, 0)
# plt.imshow(image);


# ## Do Inference
time.sleep(2.0)

# ie.infer_request.wait()
# print(f"\n{compiled_model}\n")

result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)


# Convert the inference result to a class name.
imagenet_classes = open("../data/datasets/imagenet/imagenet_2012.txt").read().splitlines()

# The model description states that for this model, class 0 is a background.
# Therefore, a background must be added at the beginning of imagenet_classes.
imagenet_classes = ['background'] + imagenet_classes

print(imagenet_classes[result_index])


