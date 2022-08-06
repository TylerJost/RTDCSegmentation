# RTDCSegmentation
This is a repository for segmenting suspended cells.

## Running the Model
An example python script that minimally loads the model is called "evaluateSegmentation.py". Note that initializing the model will take quite some time, but the model is very fast once initialized, averaging 26.2 ms +/- 220 us to load and segment a cell. 

[](./media/sampleSegmentation.png)
## Structure

Only the code is uploaded to Github. Final model weights and data are available upon request. 

Multiple experiments are concatenated for calibrating the final model. Annotation data is in COCO format and can easily be accessed from detectron 2. 
```
├── data
│   ├── allCells
│   │   ├── annotations
│   │   └── images
│   ├── experiment1
│   │   ├── annotations
│   │   └── images
├── output
│   ├── segmentCells
└── scripts
```
