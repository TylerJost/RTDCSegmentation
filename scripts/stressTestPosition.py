# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

# %% Get data
dataDir = '../data/allCells'

trainAnnotations = os.path.join(dataDir, 'annotations', 'train-annotated.json')
testAnnotations = os.path.join(dataDir, 'annotations', 'test-annotated.json')

imgDir = os.path.join(dataDir, 'images')

register_coco_instances("allCells_train", {}, trainAnnotations, imgDir)
# %%
imgs = DatasetCatalog['allCells_train']()
# %% Find centroids of each segmentation
centroids = []
for img in imgs:
    for annotation in img['annotations']:
        segmentation = annotation['segmentation'][0]
        x = [segmentation[i] for i in range(len(segmentation)) if i%2 == 0]
        y = [segmentation[i] for i in range(len(segmentation)) if i%2 != 0]
        n = len(x)
        centroid = [sum(x)/n, sum(y)/n]
        centroids.append(centroid)
centroidsx = [centroid[0] for centroid in centroids]
centroidsy = [centroid[1] for centroid in centroids]

plt.subplot(212)
plt.imshow(imread(img['file_name']))
plt.title('Exemplar Image')
plt.subplot(221)
plt.hist(centroidsx)
plt.title('X-Axis Centroid Distribution')
plt.subplot(222)
plt.hist(centroidsy)
plt.title('Y-Axis Centroid Distribution')
plt.show()
# %%
