# rootpath=$HOME/VisualSearch
rootpath=data
overwrite=0
trainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
testCollection=msrvtt10ktest

# for option in 11 12 21 22 31 32 41 42 51 52 61 62;
# do

case $1 in

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
100)
    config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_lightw2vmodeltest_threshold1
    ;;
101)
    config=w2vvpp_resnext101-resnet152_multispace_bow_w2v_lightw2vmodeltest_threshold5
    ;;
*)
    config=-1
    ;;
esac

gpu=$2
prefix=runs_$3

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

# done
