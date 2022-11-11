# HKU TCGA SARC PROJECT
This is the Pytorch implementation for the multiple instance learning model used for our TCGA SARC project.

## Processing raw WSI data
If you are processing WSI from raw images, you will need to download the WSIs first.

1. Place WSI files as `.\WSI\[DATASET_NAME]\[CATEGORY_NAME]\[SLIDE_FOLDER_NAME] (optional)\SLIDE_NAME.svs`. 
> For binary classifier, the negative class should have `[CATEGORY_NAME]` at index `0` when sorted alphabetically. For multi-class classifier, if you have a negative class (not belonging to any of the positive classes), the folder should have `[CATEGORY_NAME]` at **the last index** when sorted alphabetically. The naming of the class folders does not matter if you do not have a negative class.

> In this project, we use the combined_clinical.tsv to place the WSI.
```
  $ python ./WSI_PROCESSING/WSI_organize.py
  $ python ./LMS_Uterus_organize.py
```

2. Crop patches.  
```
  $ python deepzoom_tiler.py -m 0 -b 20 -d [DATASET_NAME]
```
>Set flag `-m [LEVEL 1] [LEVEL 2]` to crop patches from multiple magnifications. 

3. Generate the thumbnail
```
  $ python ./WSI_processing/generate_thumbnail.py
```

4. Re-name and organize the patches as graph
```
Thumbnail:"-1"
5x:"x1-x2-y1-y2-x-y",
    x/y are the coordinates of the 5x node, 
    and x1/x2/y1/y2 are the coordinates of the corresponding 10x node
10x:"x-y"
```
```
  $ python ./WSI_processing/rename_patch_as_graph_5x.py
```
```
  $ python ./WSI_processing/rename_patch_as_graph_10x.py
```

5. KimiaNet feature Extraction
```
  $ python ./WSI_processing/KimiaNet_PyTorch_Feature_Extraction.py
```

6. generate graph data
```
  $ python ./github_pretreat.py
```

7. run experiment
```
  $ python ./main_patient_wise_split.py
```

## Folder structures
Data is organized in `WSI` and `datasets`. `WSI` folder contains the images.
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- CLASS_1
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
|   |   |-- CLASS_2
|   |   |   |-- SLIDE_1.svs
|   |   |   |-- ...
```
Once patch extraction is performed with `deepzoom_tiler.py`, `sinlge` folder or `pyramid` folder will appear.
```
root
|-- WSI
|   |-- DATASET_NAME
|   |   |-- single
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
|   |   |-- pyramid
|   |   |   |-- CLASS_1
|   |   |   |   |-- SLIDE_1
|   |   |   |   |   |-- PATCH_LOW_1
|   |   |   |   |   |   |-- PATCH_HIGH_1.jpeg
|   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- PATCH_LOW_1.jpeg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
```


