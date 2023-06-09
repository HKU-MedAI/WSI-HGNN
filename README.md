This repository provides the Pytorch implementations for "Histopathology Whole Slide Image Analysis with Heterogeneous Graph Representation Learning"

Paper can be found [here](https://openaccess.thecvf.com/content/CVPR2023/html/Chan_Histopathology_Whole_Slide_Image_Analysis_With_Heterogeneous_Graph_Representation_Learning_CVPR_2023_paper.html) and video walkthrough is [here](https://youtu.be/F47ureXZ7fo).

# Download the WSIs

The WSIs can be found in the TCGA project:

https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga

# Patch Extraction

To extract the patches from the downloaded WSIs, users need to first modify the parameters in get_patches.py (including the WSI paths) and extract the patches by running the following command:

``` 
python get_patches.py
```
# Graph Construction

After the patch extraction is finished, users can obtain homogeneous and heterogeneous graphs by first edit the configurations in ./configs/GraphConstruction, and specify the correct yaml configuration file in get_graph.py, then run the following command

```
python get_graph.py
```

# Training HEAT Model

The configurations yaml files for each benchmarking dataset is grouped in respective subfolders. Users may first modify the respective config files for hyper-parameter settings, and update the path to training config in main.py.

```
python main.py
```

The training pipeline is mainly written in ./trainer/train_gnn.py. Evaluation is performed after every epoch on validation sets and testing sets. The codes can be find in ./evaluator/eval_homo_graph.py.

# Load checkpoints

The trained checkpoints will be saved in ./chekpoints, including the GNN model. Users can perform evaluation using the saved weights inside the checkpoint.
