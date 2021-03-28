
if [ "$#" -ne 1 ]; then
    echo "usage: $0 trainCollection"
fi

train_collection=$1
#train_collection='msrvtt10ktrain'

overwrite=0
root_path='data'

for encoding in bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding --overwrite $overwrite \
	--rootpath $root_path
done


