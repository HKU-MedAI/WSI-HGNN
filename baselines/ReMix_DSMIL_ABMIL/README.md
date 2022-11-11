# [MICCAI22] ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification

This repository holds the Pytorch implementation for the ReMix augmentation described in the paper 
> [**ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**](https://arxiv.org/abs/2207.01805),  
> Jiawei Yang, Hanbo Chen, Yu Zhao, Fan Yang,  Yao Zhang, Lei He, and Jianhua Yao    
> International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2022 



<p align="center">
  <img src="overview.png" width="1000">
</p>


# Installation

We use [DSMIL](https://github.com/binli123/dsmil-wsi) as the original codebase, and [mmselfsup](https://github.com/open-mmlab/mmselfsup) for contrastive learning pre-training. You can refer to their repos for installation.

# Data Preparation 
We use two datasets in our paper for demonstration: 1) [Camelyon16](https://camelyon16.grand-challenge.org/) dataset and 2) [UniToPatho](https://ieee-dataport.org/open-access/unitopatho) dataset. 


## Camelyon16
For Camelyon16 dataset, we use the pre-computed features provided by [DSMIL](https://github.com/binli123/dsmil-wsi). You can follow their instructions to download the pre-computed features. We also provide the download script (same as theirs). To download, run:

```shell
python3 tools/process_dataset.py --dataset Camelyon16 --task download
```

The pre-computed features are provided as csv files (.csv). Please consider converting the csv files to numpy array files (.npy). In our machine, this step improves the original DSMIL/ABMIL training by 2X faster speed and 2.4X less memory consumption. To convert, run:

```shell
python3 tools/process_dataset.py --dataset Camelyon16 --task convert
```

Note that, the published training code in DSMIL re-splits Camelyon16 into 80%/20% for training and testing. In contrast, we use the official training/testing split from Camelyon16. To do so, run:

```shell
python3 tools/process_dataset.py --dataset Camelyon16 --task split
```

Or you can put all the commands together as:

```shell
python3 tools/process_dataset.py --dataset Camelyon16 --task download convert split
```
## UniToPatho

### Processing

Following instructions in [unitopatho](https://ieee-dataport.org/open-access/unitopatho) to download the original dataset. We use the data in the `800` folder. To prepare datasets for contrastive learning, we first crop UniToPatho images for pre-training and downstream MIL tasks, and gather related meta files. To do so, run:

```shell
python3 tools/process_dataset.py --dataset UniToPatho --task crop split
```

### Self-supervised Learning

For self-supervised learning, we use the open-source [mmselfsup](https://github.com/open-mmlab/mmselfsup) toolbox for SimCLR pre-training. Follow their instructions for toolbox installation and usage. Note that, the `mmselsup` toolbox had been updated several times. Therefore, we provide the version we used in [OpenSelfSup-MIL](../OpenSelfSup-MIL).

To use OpenSelfSup-MIL, you should first link the dataset path, and run:

```shell
cd OpenSelfSup-MIL
mkdir data
ln -s  ../datasets/Unitopatho data/Unitopatho
```

Then, everything should be clear following [README.md](../OpenSelfSup-MIL/README.md) there.

We provide the training [config file](../OpenSelfSup-MIL/configs/wsi/Unitopatho/simclr_r18_bs512_ep200.py) for our contrastive learning pre-training.

To run pre-training in an 8-GPU machine, run:

```shell
cd OpenSelfSup-MIL
bash tools/dist_train.sh configs/wsi/Unitopatho/simclr_r18_bs512_ep200.py 8
```

### Feature extraction

After pre-training, run the following command to extract features for downstream MIL tasks:

```shell
cd OpenSelfSup-MIL

ckpt_pth=work_dirs/wsi/Unitopatho/simclr_r18_bs512_ep200

python3 tools/extract_backbone_weights.py ${ckpt_pth}/epoch_200.pth ${ckpt_pth}/extracted_weights_ep200.pth

python3 tools/extract_feats_unitopatho.py \
        --pretrain ${ckpt_pth}/extracted_weights_ep200.pth \
        --config configs/wsi/extraction_config.py 
```

The extracted features are saved in the `OpenSelfSup-MIL/data/Unitopatho/features` folder, which is linked from `dataset/Unitopatho/features`.



# Reduce

To reduce the number of instances per bag, ReMix uses KMeans clustering to select the most representative instances. We use the [faiss](https://github.com/facebookresearch/faiss) KMeans implementation. Run:

```shell
# Camelyon16
python3 reduce.py --dataset Camelyon16 --num_prototypes 8
# UniToPatho
python3 reduce.py --dataset Unitopatho --num_prototypes 1
```

You can further control the number of generated semantic shift vectors by passing, e.g., `--num_shift_vectors 500`.



# Training ReMix
To train remix, run:

```shell
python3 train_remix.py \
        --dataset Camelyon16 \
        --model dsmil \ 
        --num_prototypes 8 \
        --mode cov \
        --rate 0.5 \
        --exp_name k8_aug_cov 

python3 train_remix.py \
        --dataset Unitopatho \
        --model dsmil \
        --num_prototypes 1 \ 
        --mode cov \
        --rate 0.5 \
        --exp_name dsmil_k1_aug_cov 
```
You can specify augmentation mode (`--mode`, choose among [None, replace, append, interpolate, cov, and joint]), and augmentation probability (`--rate`, float number between [0, 1]). For `joint` augmentation, we recommend a lower rate (e.g., 0.2 or 0.1) than our default rate of 0.5.


# Disclaimer
This tool is for research purposes and is not approved for clinical use.

This is not an official Tencent product.

# Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.

# Citation
Please consider citing our paper in your publications if the project helps your research.



