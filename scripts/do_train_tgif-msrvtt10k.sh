rootpath=$HOME/VisualSearch
overwrite=0

trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA

if [ "$#" -ne 2 ];then
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

CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection  --overwrite $overwrite\
    --rootpath $rootpath --config $config --val_set $val_set --model_prefix $prefix 