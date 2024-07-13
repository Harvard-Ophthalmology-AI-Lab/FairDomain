#!/bin/bash
# DATASET_DIR=/data/home/shim/pyspace/others/pyspace/ICLR_30k/ODIR_Data/ODIR_Data_New_seed10
DATASET_DIR=/home/shim/pyspace/CDTrans/data/FairDomain_Classify/npzs_enhance_224
RESULT_DIR=./IRM_results
MODEL_TYPE=( efficientnet ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
NUM_EPOCH=20
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

# OPTIMIZER='adamw'
# OPTIMIZER_ARGUMENTS='{"lr": 0.001, "weight_decay": 0.01}'

# SCHEDULER='step_lr'
# SCHEDULER_ARGUMENTS='{"step_size": 30, "gamma": 0.1}'

if [ ${MODALITY_TYPE} = 'oct_bscans' ]; then
	LR=1e-4 
	BATCH_SIZE=6
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=1e-4
# 	LR=( 1e-3 1e-4 1e-5 1e-6 1e-7 )
	BATCH_SIZE=10
else
	LR=1e-4
	BATCH_SIZE=6
fi


BLR=5e-4
WD=6e-5 # 0.0001
LD=0.55
DP=0.1
VIT_WEIGHTS=imagenet

NEED_BALANCED=False
# VIT_WEIGHT=dinov2
MODEL_TYPE=( swin ) # ( efficientnet densenet resnet swin convnext ViT-B vgg )
ATTRIBUTE_TYPE=( race )

for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
do

NUM_EPOCH=10
BATCH_SIZE=10
PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}.csv

if [ ${MODEL_TYPE[$j]} = 'ViT-B' ]; then
    # NEED_BALANCED=True
    if [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
        BATCH_SIZE=64
    fi
    WD=0.01
    NUM_EPOCH=20
    LR=1e-4
    BLR=5e-4
fi

if [ ${MODEL_TYPE[$j]} = 'vgg' ]; then
    # NEED_BALANCED=True
    LR=1e-5
    WD=0.0
    BATCH_SIZE=10
    NUM_EPOCH=20
fi

if [ ${MODEL_TYPE[$j]} = 'efficientnet' ]; then
    # NEED_BALANCED=True
    LR=1e-4
    WD=0.01
    BATCH_SIZE=10
fi

if [ ${MODEL_TYPE[$j]} = 'swin' ]; then
    # NEED_BALANCED=True
    LR=1e-4
    WD=0.01
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'densenet' ]; then
    # NEED_BALANCED=True
    LR=5e-4
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

python ./scripts/train_IRM.py \
		--data_dir ${DATASET_DIR}/ \
		--result_dir ${RESULT_DIR}/results_IRM/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}/${MODEL_TYPE[$j]}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE} \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 224 \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch_size ${BATCH_SIZE} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE} \
        --attribute_type ${ATTRIBUTE_TYPE[$a]} \
        --vit_weights ${VIT_WEIGHTS} \
        --blr ${BLR} \
        --drop_path ${DP} \
        --layer_decay ${LD} \
        --weight_decay ${WD} \
        --need_balance False
done
done
