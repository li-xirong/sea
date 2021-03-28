rootpath=data
overwrite=0

# if [ "$#" -ne 3 ]; then
#     echo "Usage: $0 testCollection topic_set sim_name"
#     exit
# fi

trainCollection=tgif-msrvtt10k
valCollection=tv2016train
val_set=setA

config=$1
prefix=runs_0

model_path=$rootpath/$trainCollection/w2vvpp_train/$valCollection/$val_set/$config/$prefix/model_best.pth.tar
sim_name=$trainCollection/$valCollection/$val_set/$config/$prefix

for topic_set in {tv16,tv17,tv18}; do

    test_collection=iacc.3
    score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
    echo $score_file

    bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
    python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite
done

for topic_set in {tv19,tv20}; do
    test_collection=v3c1

    score_file=$rootpath/$test_collection/SimilarityIndex/$topic_set.avs.txt/$sim_name/id.sent.score.txt
    echo $score_file

    bash do_txt2xml.sh $test_collection $score_file $topic_set $overwrite
    python trec_eval.py ${score_file}.xml --rootpath $rootpath --collection $test_collection --edition $topic_set --overwrite $overwrite
done
