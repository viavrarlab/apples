# see https://github.com/facebookresearch/d2go/blob/main/demo/d2go_beginner.ipynb


import os
import json
import random


import cv2 as cv
import torch
from mobile_cv.predictor.api import create_predictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import STABLE_ONNX_OPSET_VERSION
from d2go.model_zoo import model_zoo
from d2go.runner import GeneralizedRCNNRunner
from d2go.export.exporter import convert_and_export_predictor
from d2go.utils.testing.data_loader_helper import create_detection_data_loader_on_toy_dataset
from d2go.utils.demo_predictor import DemoPredictor


from py import hook


setup_logger()


DATASETS = ("apples.v2i.coco", "detection_plant.v1i.coco")
SUBSETS = ("train", "valid")
THING_CLASSES = ["fresh-apple", "damaged-apple", "weed"]
CFG_NAME = "qat_faster_rcnn_fbnetv3a_C4"
ADD_SEGMENTATION = True
TRAIN = True
TEST = True
EXPORT = True and not TRAIN


for dataset in DATASETS:
    for subset in SUBSETS:
        path = f"data/{dataset}/{subset}/_annotations.coco.json"
        if not os.path.isfile(path):
            continue
        name = f"apples_{dataset}_{subset}"
        if ADD_SEGMENTATION:
            with open(path, "r+") as file:
                data = json.load(file)
                for annotation in data["annotations"]:
                    x, y, w, h = annotation["bbox"]
                    annotation["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                file.seek(0)
                file.truncate()
                json.dump(data, file, ensure_ascii=False, indent=4, sort_keys=True)
        register_coco_instances(name, {}, path, f"data/{dataset}/{subset}")
        MetadataCatalog.get(name).set(thing_classes=THING_CLASSES, evaluator_type="coco")


class Runner(GeneralizedRCNNRunner):
    def _get_trainer_hooks(self, cfg, model, optimizer, scheduler, periodic_checkpointer, trainer):
        hooks = super()._get_trainer_hooks(cfg, model, optimizer, scheduler, periodic_checkpointer, trainer)
        hooks.append(hook.LossEvalHook(cfg, model, build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], DatasetMapper(cfg, True)), periodic_checkpointer))
        return hooks


runner = Runner()
cfg = runner.get_default_cfg()
cfg.merge_from_file(f"configs/{CFG_NAME}.yaml")
cfg.OUTPUT_DIR = f"output/{CFG_NAME}"
cfg.DATASETS.TRAIN = tuple(f"apples_{dataset}_train" for dataset in DATASETS)
cfg.DATASETS.TEST = tuple(f"apples_{dataset}_valid" for dataset in DATASETS)


cfg.MODEL.DEVICE = "cuda"
cfg.MODEL_EMA.ENABLED = False
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(THING_CLASSES)
cfg.TEST.EVAL_PERIOD = 200


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
if os.path.isfile(cfg.MODEL.WEIGHTS):
    with open(cfg.MODEL.WEIGHTS, "r") as file:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, file.read().strip())
resume = os.path.isfile(cfg.MODEL.WEIGHTS)
if not resume:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{CFG_NAME[4:] if CFG_NAME.startswith('qat_') else CFG_NAME}.yaml")


os.makedirs(os.path.join(cfg.OUTPUT_DIR, "predictor"), exist_ok=True)


if TRAIN:
    model = runner.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    runner.do_train(cfg, model, resume=resume)
    metrics = runner.do_test(cfg, model)
    print(metrics)


if EXPORT:
    model = runner.build_model(cfg, True).to("cpu")
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    with create_detection_data_loader_on_toy_dataset(cfg, 224, 320, is_train=False) as loader:
        path = convert_and_export_predictor(cfg, model, "torchscript_int8", os.path.join(cfg.OUTPUT_DIR, "predictor"), loader)
        image = next(iter(loader))[0]["image"]
    model = torch.jit.load(os.path.join(path, "model.jit"))
    torch.onnx.export(
        model, image, os.path.join(cfg.OUTPUT_DIR, "model.onnx"),
        input_names=("image",), output_names=("bbox", "class", "confidence", "size"),
        verbose=True, opset_version=STABLE_ONNX_OPSET_VERSION
    )


if EXPORT and TEST:
    predictor = DemoPredictor(create_predictor(path))
    cfg.DATASETS.TEST = tuple(f"apples_{dataset}_valid" for dataset in DATASETS)
    metrics = runner.do_test(cfg, predictor.model)
    print(metrics)
    for dataset in DATASETS:
        apples_dataset = DatasetCatalog.get(f"apples_{dataset}_valid")
        apples_metadata = MetadataCatalog.get(f"apples_{dataset}_valid")
        for apple in random.sample(apples_dataset, 5):
            img = cv.imread(apple["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=apples_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
            outputs = predictor(img)
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv.imshow("apple", out.get_image()[:, :, ::-1])
            cv.waitKey(10000)
    cv.destroyAllWindows()
