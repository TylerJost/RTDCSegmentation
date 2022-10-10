# %%
import json
import os
import random
import shutil
import numpy as np
 # %% Try to make new json files
def splitCOCO(originalFile, percentTrain=0.75, seed=1234):
    """
    Takes a COCO-formatted JSON file and splits into a train and test dataset
    
    Input:
    originalFile: The initial .json file with annotations and images
    percentTrain: The percentage of images to train on
    seed: Random seed for shuffling dataset
    
    Output:
    
    
    """
    # Load data
    f = open(originalFile)
    cocoData = json.load(f)
    f.close()
    # Parse out data from json file
    cocokeys = list(cocoData.keys())

    licenses =      cocoData[cocokeys[0]]
    info =          cocoData[cocokeys[1]]
    categories =    cocoData[cocokeys[2]]
    images =        cocoData[cocokeys[3]]
    annotations =   cocoData[cocokeys[4]]
    # Initialize new dictionaries in same format
    trainCoco = {}
    trainCoco['licenses'] = licenses.copy()
    trainCoco['info'] = info.copy()
    trainCoco['categories'] = categories.copy()
    trainCoco['annotations'] = []
    trainCoco['images'] = []

    testCoco = {}
    testCoco['licenses'] = licenses.copy()
    testCoco['info'] = info.copy()
    testCoco['categories'] = categories.copy()
    testCoco['annotations'] = []
    testCoco['images'] = []

    # There are more annotations than images (TODO: Follow up with Christian on this)
    # Also they are in order but might not always be
    imgIds = [im['id'] for im in images]
    imgID = {imID:img for imID, img in zip(imgIds, images)}

    # Find the number to put in train and test
    # Use a random permutation (with seed)
    nTrain = int(percentTrain*len(annotations))
    random.seed(seed)
    randomizedIndex = random.sample( range(len(annotations)), len(annotations))

    # Store new data
    c = 0
    for idx in randomizedIndex:
        annotation = annotations[idx]
        image_id = annotation['image_id']
        if c<=nTrain:
            trainCoco['annotations'].append(annotation)
            trainCoco['images'].append(imgID[image_id])
        else:
            testCoco['annotations'].append(annotation)
            testCoco['images'].append(imgID[image_id])        
        c+=1
    fileDir = '/'.join(originalFile.split('/')[0:-1])

    # Write files
    with open(os.path.join(fileDir,"train.json"), "w") as outfile:
        json.dump(trainCoco, outfile)

    with open(os.path.join(fileDir,"test.json"), "w") as outfile:
        json.dump(testCoco, outfile)
    print('Finished writing new JSON files')
# %%
percentTrain = 0.75
seed = 1234
originalFile = '../data/allCells/annotations/instances_default.json'
splitCOCO(originalFile)
# %%
