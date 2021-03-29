
# rootpath=$HOME/VisualSearch
rootpath=data
overwrite=0

trainCollection=msvdtrain
valCollection=msvdval
testCollection=msvdtest

config=$1
gpu=$2

prefix=runs_0
model_path=$rootpath/$trainCollection/sea_train/$valCollection/$config/$prefix/model_best.pth.tar
sim_name=$trainCollection/sea_train/$valCollection/$config/$prefix

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection $model_path $sim_name \
    --query_sets $testCollection.caption.txt \
    --rootpath $rootpath  --overwrite $overwrite