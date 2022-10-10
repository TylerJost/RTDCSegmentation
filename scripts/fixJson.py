# %%
import json
# %%
dataDir = '../data/allCells'

trainJson = os.path.join(dataDir, 'annotations', 'train.json')
imgDir = os.path.join(dataDir, 'images')

with open(trainJson) as f:
    data = json.load(f)

print(len(data['annotations']))

categoryDict = {'mcf10a': 1, 'mda_mb_231': 2, 'mcf7': 3}
for idx, img in enumerate(data['annotations']):
    if img['category_id'] == 2:
        print('got rid of empty image')
        data['annotations'].pop(idx)
        data['images'].pop(idx)

for img in data['annotations']:
    cellLine = img['image_id'].split('-')[-1]
    img['category_id'] = categoryDict[cellLine]

data['categories'] = [{'id': 1, 'name': 'mcf10a', 'supercategory': ''}, \
                      {'id': 2, 'name': 'mda_mb_231', 'supercategory': ''}, \
                      {'id': 3, 'name': 'mcf7', 'supercategory': ''}]

trainJsonNew = os.path.join(dataDir, 'annotations', 'train-annotated.json')
with open(trainJsonNew, "w") as outfile:
    json.dump(data, outfile)
# %%
testJson = os.path.join(dataDir, 'annotations', 'test.json')
imgDir = os.path.join(dataDir, 'images')

with open(testJson) as f:
    data = json.load(f)

print(len(data['annotations']))
allIds = []
allLines = []

categoryDict = {'mcf10a': 1, 'mda_mb_231': 2, 'mcf7': 3}
for idx, img in enumerate(data['annotations']):
    if img['category_id'] == 2:
        print('got rid of empty image')
        data['annotations'].pop(idx)
        data['images'].pop(idx)

for img in data['annotations']:
    cellLine = img['image_id'].split('-')[-1]
    allLines.append(cellLine)
    img['category_id'] = categoryDict[cellLine]

data['categories'] = [{'id': 1, 'name': 'mcf10a', 'supercategory': ''}, \
                      {'id': 2, 'name': 'mda_mb_231', 'supercategory': ''}, \
                      {'id': 3, 'name': 'mcf7', 'supercategory': ''}]

testJsonNew = os.path.join(dataDir, 'annotations', 'test-annotated.json')
with open(testJsonNew, "w") as outfile:
    json.dump(data, outfile)