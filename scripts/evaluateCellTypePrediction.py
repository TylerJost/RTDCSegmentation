# %%
import torch
import detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random, pickle
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

# %%
os.listdir('../data')

dataDir = '../data/allCells'

trainAnnotations = os.path.join(dataDir, 'annotations', 'train-annotated.json')
testAnnotations = os.path.join(dataDir, 'annotations', 'test-annotated.json')

imgDir = os.path.join(dataDir, 'images')

# %%
from detectron2.data.datasets import register_coco_instances
register_coco_instances("allCells_train", {}, trainAnnotations, imgDir)
register_coco_instances("allCells_test", {}, testAnnotations, imgDir)

# %%
# datasetDicts = DatasetCatalog['allCells_train']()
# random.seed(1234)
# for d in random.sample(datasetDicts, 5):
#     print(d["file_name"])
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('allCells_train'), scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     plt.imshow(out.get_image()[:, :, ::-1])
#     plt.show()

# %%
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("allCells_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../output/segmentCells'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
predictor = DefaultPredictor(cfg)

# %%
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator('allCells_test', output_dir='./output/segmentCells')
val_loader = build_detection_test_loader(cfg, "allCells_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# %% Evaluate num correct
dataset_dicts = DatasetCatalog['allCells_test']()
# %%
lines = []
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  
    predType = outputs['instances'].pred_classes.item()
    lines.append(predType)
# %%
nIms = 1
imSpace = 1
lines = []
for d in random.sample(dataset_dicts, nIms): 
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get('allCells_test'),
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.title(d['file_name'])
    print(outputs['instances'].pred_classes)

# %% Find accuracy
predicted = []
actual = []
scores = []
correct = 0
cellDict = {'mcf10a': 1, 'mda_mb_231': 2, 'mcf7': 3}
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])    
    cellLine = d["file_name"].split('.tif')[0].split('-')[-1]
    outputs = predictor(im)
    bestPred = np.argmax(outputs['instances'].scores)
    scores.append(outputs['instances'].scores[bestPred])
    predictedVal = outputs['instances'].pred_classes[bestPred].item()+1
    actualVal = cellDict[cellLine]
    predicted.append(predictedVal)
    actual.append(actualVal)
    if predictedVal == actualVal:
        correct += 1
print(f'Accuracy: {correct/len(dataset_dicts):0.2f}')