import sys

sys.path.append("../yolov5")

from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import check_dataset, non_max_suppression, scale_coords, xywh2xyxy, check_yaml,increment_path
from yolov5.utils.metrics import ap_per_class
from yolov5.val import process_batch

from openvino.tools.pot.api import Metric, DataLoader
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.utils.logger import init_logger, get_logger

from pathlib import Path
from typing import Tuple, Dict

def get_config():
    """ Set the configuration of the model, engine, 
    dataset, metric and quantization algorithm.
    """
    config = dict()
    data_yaml = check_yaml("./yolov5/data/coco128.yaml")
    data = check_dataset(data_yaml)

    model_config = Dict({
        "model_name": "yolov5m",
        "model": "./yolov5/yolov5m/yolov5m_openvino_model/yolov5m.xml",
        "weights": "./yolov5/yolov5m/yolov5m_openvino_model/yolov5m.bin"
    })

    engine_config = Dict({
        "device": "CPU",
        "stat_requests_number": 8,
        "eval_requests_number": 8
    })

    dataset_config = Dict({
        "data_source": data,
        "imgsz": 640,
        "single_cls": True,
    })

    metric_config = Dict({
        "conf_thres": 0.001,
        "iou_thres": 0.65,
        "single_cls": True,
        "nc": 1 ,  # if opt.single_cls else int(data['nc']),
        "names": data["names"],
        "device": "cpu"
    })

    algorithms = [
        {
            "name": "DefaultQuantization",  # or AccuracyAware
            "params": {
                    "target_device": "CPU",
                    "preset": "mixed",
                    "stat_subset_size": 300
            }
        }
    ]

    config["model"] = model_config
    config["engine"] = engine_config
    config["dataset"] = dataset_config
    config["metric"] = metric_config
    config["algorithms"] = algorithms
    
    return config


class YOLOv5DataLoader(DataLoader):
    """ Inherit from DataLoader function and implement for YOLOv5.
    """

    def __init__(self, config):
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)

        self._data_source = config.data_source
        self._imgsz = config.imgsz
        self._batch_size = 1
        self._stride = 32
        self._single_cls = config.single_cls
        self._pad = 0.5
        self._rect = False
        self._workers = 1
        self._data_loader = self._init_dataloader()
        self._data_iter = iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader.dataset)

    def _init_dataloader(self):
        dataloader = create_dataloader(self._data_source['val'], imgsz=self._imgsz, batch_size=self._batch_size, stride=self._stride,
                                       single_cls=self._single_cls, pad=self._pad, rect=self._rect, workers=self._workers)[0]
        return dataloader

    def __getitem__(self, item):
        try:
            batch_data = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch_data = next(self._data_iter)

        im, target, path, shape = batch_data

        im = im.float()  
        im /= 255  
        nb, _, height, width = im.shape  
        img = im.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        annotation = dict()
        annotation['image_path'] = path
        annotation['target'] = target
        annotation['batch_size'] = nb
        annotation['shape'] = shape
        annotation['width'] = width
        annotation['height'] = height
        annotation['img'] = img

        return (item, annotation), img

# ----- #

""" Download dataset and set config
"""
print("Run the POT. This will take few minutes...")
config = get_config()  
init_logger(level='INFO')
logger = get_logger(__name__)
save_dir = increment_path(Path("./yolov5/yolov5m/yolov5m_openvino_model/"), exist_ok=True)  # increment run
save_dir.mkdir(parents=True, exist_ok=True)  # make dir

# Step 1: Load the model.
model = load_model(config["model"])

# Step 2: Initialize the data loader.
data_loader = YOLOv5DataLoader(config["dataset"])


# Step 4: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=config["engine"], data_loader=data_loader, metric=metric)

# Step 5: Create a pipeline of compression algorithms.
pipeline = create_pipeline(config["algorithms"], engine)

metric_results = None

# Check the FP32 model accuracy.
metric_results_fp32 = pipeline.evaluate(model)

logger.info("FP32 model metric_results: {}".format(metric_results_fp32))

# Step 6: Execute the pipeline to calculate Min-Max value
compressed_model = pipeline.run(model)

# Step 7 (Optional):  Compress model weights to quantized precision
#                     in order to reduce the size of final .bin file.
compress_model_weights(compressed_model)

# Step 8: Save the compressed model to the desired path.
optimized_save_dir = Path(save_dir).joinpath("optimized")
save_model(compressed_model, Path(Path.cwd()).joinpath(optimized_save_dir), config["model"]["model_name"])

# Step 9 (Optional): Evaluate the compressed model. Print the results.
metric_results_i8 = pipeline.evaluate(compressed_model)

logger.info("Save quantized model in {}".format(optimized_save_dir))
logger.info("Quantized INT8 model metric_results: {}".format(metric_results_i8))