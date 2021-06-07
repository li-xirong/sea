rootpath=$HOME/VisualSearch
overwrite=0

trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA
testCollection=iacc.3
query_sets=tv16.avs.txt,tv17.avs.txt,tv18.avs.txt

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 config gpu_id"
    exit
fi

config=$1
prefix=runs_0

model_path=$rootpath/$trainCollection/Models/$valCollection/$val_set/$config/$prefix/model_best.pth.tar
sim_name=$trainCollection/$valCollection/$val_set/$config/$prefix

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

gpu=$2
CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection $model_path $sim_name \
    --query_sets $query_sets \
    --rootpath $rootpath \
    --overwrite $overwrite 