
train_collection='tgif-msrvtt10k-vatex'
overwrite=0

for encoding in bow bow_nsw gru
do
    python build_vocab.py $train_collection --encoding $encoding --overwrite $overwrite
done


