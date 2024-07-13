model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
for target_dataset in 'slo_fundus'
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR './logs/gender/'$model'/fairdomain/oct_fundus'$target_dataset \
    MODEL.PRETRAIN_PATH './logs/pretrain/'$model'/fairdomain/Fundus/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/FairDomain_Classify/oct_fundus/train/oct_train_list.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/FairDomain_Classify/slo_fundus/test/slo_test_list.txt' \
    DATASETS.ROOT_TEST_DIR './data/FairDomain_Classify/slo_fundus/test/slo_test_list.txt' \
    DATASETS.NAMES "Fundus" DATASETS.NAMES2 "Fundus" \
    MODEL.Transformer_TYPE $model_type \
    SOLVER.LOG_PERIOD 10 \
    
 
done
