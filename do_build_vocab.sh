overwrite=0
rootpath=$HOME/VisualSearch

if [ "$#" -ne 1 ]; then
    echo "usage: $0 trainCollection"
fi

train_collection=$1

for encoding in bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding --overwrite $overwrite \
	--rootpath $rootpath
done