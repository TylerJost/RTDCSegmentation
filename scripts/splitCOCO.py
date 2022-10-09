# %%
import json
import os
import random
import shutil

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
    for idx in randomizedIndex:
        annotation = annotations[idx]
        image_id = annotation['image_id']
        if idx<=nTrain:
            trainCoco['annotations'].append(annotation)
            trainCoco['images'].append(imgID[image_id])
        else:
            testCoco['annotations'].append(annotation)
            testCoco['images'].append(imgID[image_id])        

    fileDir = '/'.join(originalFile.split('/')[0:-1])

    # Write files
    with open(os.path.join(fileDir,"train.json"), "w") as outfile:
        json.dump(trainCoco, outfile)

    with open(os.path.join(fileDir,"test.json"), "w") as outfile:
        json.dump(testCoco, outfile)
    print('Finished writing new JSON files')
    
percentTrain = 0.75
seed = 1234
originalFile = '../data/allCells/annotations/instances_default.json'
splitCOCO(originalFile)

# %% [markdown]
# # Combine COCO Datasets

# %%
# def combineCOCO(cocolist):

experiments = ['mcf7', 'mcf10a', 'mda_mb_231']


# mcf7Annotated = '../data/mcf7/annotations/instances_default.json'
# mcf10aAnnotated = '../data/mcf10a/annotations/instances_default.json'
# mda_mb_231Annotated = '../data/mda_mb_231/annotations/instances_default.json'

concatBase = '../data/allCells'
concatIms = os.path.join(concatBase, 'images')
concatAnnotations = os.path.join(concatBase, 'annotations')
allImages = []
allAnnotations = []

for experiment in experiments:
    file = os.path.join('../data', experiment, 'annotations','instances_default.json')
    # Load data
    f = open(file)
    cocoData = json.load(f)
    f.close()
    # Parse out data from json file
    cocokeys = list(cocoData.keys())

    licenses =      cocoData[cocokeys[0]]
    info =          cocoData[cocokeys[1]]
    categories =    cocoData[cocokeys[2]]
    images =        cocoData[cocokeys[3]]
    annotations =   cocoData[cocokeys[4]]
    
    # Add annotation to both id, image id, and copy over image
    
    for annotation in annotations:
        annotation['id'] = str(annotation['id'])+'-'+experiment
        annotation['image_id'] = str(annotation['image_id'])+'-'+experiment
    for image in images:
        image['id'] = str(image['id'])+'-'+experiment
        # Copy and rename image
        imgName = image['file_name']    
        image['file_name'] = imgName.split('.tif')[0]+'-'+experiment+'.tif'
        
        imgLocation = os.path.join('../data', experiment, 'images', imgName)
        imgLocationNew = os.path.join(concatIms, image['file_name'])
        print('Copying {} to {}'.format(imgLocation, imgLocationNew))
        shutil.copy(imgLocation, imgLocationNew)
    allImages += images
    allAnnotations += annotations

# Write concatenated data
cocoData = {cocokeys[0]:licenses, cocokeys[1]:info, cocokeys[2]:categories, 
           cocokeys[3]: allImages, 
           cocokeys[4]: allAnnotations}

with open(os.path.join(concatAnnotations,"instances_default.json"), "w") as outfile:
    json.dump(cocoData, outfile)


# %%

# %%
mcf7 = '../data/mcf7/annotations/instances_default.json'
mcf10a = '../data/mcf10a/annotations/instances_default.json'
mda_mb_231 = '../data/mda_mb_231/annotations/instances_default.json'

# %%
splitCOCO(mcf7)
splitCOCO(mcf10a)
splitCOCO(mda_mb_231)



# %%
appendID(mcf7, 'mcf7-1')
appendID(mcf10a, 'mcf10a-1')
appendID(mda_mb_231, 'mda_mb_231-1')

# %%

# %%
[1,2,3]+[3,4,5]

# %%
