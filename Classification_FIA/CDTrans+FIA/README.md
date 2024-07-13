# CDTrans with Fair Identity Attention (CDTrans+FIA)

## Introduction

We adapt their official code from CDTrans and plug into our FIA module

## Requirements
### Installation
```bash
pip install -r requirements.txt
(Python version is the 3.7 and the GPU is the V100 with cuda 10.1, cudatoolkit 10.1)
```
### Prepare Datasets
Download the FairDomain datasets.

Then unzip them and rename them under the directory like follow:


### Prepare DeiT-trained Models
For fair comparison in the pre-training data set, we use the DeiT parameter init our model based on ViT. 
You need to download the ImageNet pretrained transformer model : [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth) and move them to the `./data/pretrainModel` directory.

# Scripts.

```bash

# Office-31     Source: Amazon   ->  Target: Dslr, Webcam
python train.py scripts/pretrain/office31/run_office_amazon.sh deit_base
bash scripts/uda/office31/run_office_amazon.sh deit_base

#Office-Home    Source: Art      ->  Target: Clipart, Product, Real_World
bash scripts/pretrain/officehome/run_officehome_Ar.sh deit_base
bash scripts/uda/officehome/run_officehome_Ar.sh deit_base

# VisDA-2017    Source: train    ->  Target: validation
bash scripts/pretrain/visda/run_visda.sh deit_base
bash scripts/uda/visda/run_visda.sh deit_base

# DomainNet     Source: Clipart  ->  Target: painting, quickdraw, real, sketch, infograph
bash scripts/pretrain/domainnet/run_domainnet_clp.sh deit_base
bash scripts/uda/domainnet/run_domainnet_clp.sh deit_base
```
DeiT-Small scripts
Replace deit_base with deit_small to run DeiT-Small results. An example of training on office-31 is as follows:
```bash
# Office-31     Source: Amazon   ->  Target: Dslr, Webcam
bash scripts/pretrain/office31/run_office_amazon.sh deit_small
bash scripts/uda/office31/run_office_amazon.sh deit_small
```

## Evaluation
```bash
# For example VisDA-2017
python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/vit_base/visda/transformer_best_model.pth')" DATASETS.NAMES 'VisDA' DATASETS.NAMES2 'VisDA' OUTPUT_DIR '../logs/uda/vit_base/visda/' DATASETS.ROOT_TRAIN_DIR './data/visda/train/train_image_list.txt' DATASETS.ROOT_TRAIN_DIR2 './data/visda/train/train_image_list.txt' DATASETS.ROOT_TEST_DIR './data/visda/validation/valid_image_list.txt'  
```

## Acknowledgement

Codebase from [TransReID](https://github.com/damo-cv/TransReID)


