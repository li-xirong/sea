
# rootpath=$HOME/VisualSearch
rootpath=data
overwrite=1

trainCollection=msvdtrain
valCollection=msvdval
testCollection=msvdtest

option=$1
case ${option} in
11)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v
;;
12)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v
;;
21)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v_gru
;;
22)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_gru
;;
31)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v_bigru
;;
32)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_bigru
;;
41)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v_bert
;;
42)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_bert
;;
51)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v_gru_bert
;;
52)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_gru_bert
;;
61)
config=w2vvpp_resnext101-resnet152_subspace_bow_w2v_bigru_bert
;;
62)
config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_bigru_bert
;;
*)
config=-1
esac

NO=$3
prefix=runs_$NO
model_path=$rootpath/msvdtrain/sea_train/msvdval/$config/$prefix/model_best.pth.tar
sim_name=msvdtrain/w2vvpp_train/msvdval/$config/$prefix

if [ ! -f "$model_path" ]; then
    echo "model not found: $model_path"
    exit
fi

gpu=$2
CUDA_VISIBLE_DEVICES=$gpu python predictor.py $testCollection $model_path $sim_name \
    --query_sets $testCollection.caption.txt \
    --rootpath $rootpath  --overwrite $overwrite


