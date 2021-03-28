# rootpath=$HOME/VisualSearch
rootpath=data
overwrite=0
trainCollection=msvdtrain
valCollection=msvdval
testCollection=msvdtest

if [ "$#" -ne 2];then
    echo "Usage: $0 config gpuID"
    exit
fi

# config=sea_resnext101-resnet152_bow_w2v
# config=sea_resnext101-resnet152_bow_w2v_gru
# config=sea_resnext101-resnet152_bow_w2v_bigru
# config=sea_resnext101-resnet152_bow_w2v_bert
# config=sea_resnext101-resnet152_bow_w2v_gru_bert
# config=sea_resnext101-resnet152_bow_w2v_bigru_bert

config=$1
gpu=$2

prefix=runs_0
# ---train---
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection --overwrite $overwrite --rootpath $rootpath --config $config --model_prefix $prefix
#exit
# ---test---
model_path=$rootpath/$trainCollection/sea_train/$valCollection/$config/$prefix/model_best.pth.tar
sim_name=$trainCollection/sea_train/$valCollection/$config/$prefix

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection $model_path $sim_name \
    --query_sets $testCollection.caption.txt \
    --rootpath $rootpath --overwrite $overwrite
