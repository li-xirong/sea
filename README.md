# Sea
Souce code of the papar: [SEA: Sentence Encoder Assembly for Video Retrieval by Textual Queries](https://arxiv.org/abs/2011.12091). 


![image](framework.png)

The code assumes [video-level CNN features](https://github.com/xuchaoxi/video-cnn-feat) have been extracted. 


## Environment
* Ubuntu 16.04
* cuda 10.1
* python 3.7
* PyTorch 1.4.0
* tensorboard 2.1.0
* numpy 1.19.5

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.
This is an example of creating an conda environment.
```
conda create -n sea python==3.7
conda activate sea
git clone https://github.com/li-xirong/sea.git
cd sea
pip install -r requirements.txt
```

## Required Data
### Dataset arrangement
```bash
# Use msvd as an example.
├── msvd
│   ├── FeatureData
│   │   └── mean_resnext101_resnet152
│   │       ├── feature.bin
│   │       ├── id.txt
│   │       └── shape.txt
│   └── TextData
│       ├── msvd.caption.txt
│       └── PrecomputedSentFeat
│           └── bert_feature_Layer_-2_uncased_L-12_H-768_A-12
│                ├── feature.bin
│                ├── id.txt
│                └── shape.txt
│
├── msvdtrain
│   ├── FeatureData -> ../msvd/FeatureData/
│   ├── TextData
│   │   ├── msvdtrain.caption.txt
│   │   ├── PrecomputedSentFeat -> ../../msvd/TextData/PrecomputedSentFeat/
│   └── VideoSets
│       └── msvdtrain.txt
│
├── msvdval
│   ├── FeatureData
│   │   └── mean_resnext101_resnet152 -> ../../msvd/FeatureData/mean_resnext101_resnet152/
│   ├── TextData
│   │   ├── msvdval.caption.txt
│   │   └── PrecomputedSentFeat -> ../../msvd/TextData/PrecomputedSentFeat/
│   └── VideoSets
│       └── msvdval.txt
│
└── msvdtest
     ├── FeatureData -> ../msvd/FeatureData/
     ├── TextData
     │   ├── msvdtest.caption.txt
     │   ├── PrecomputedSentFeat -> ../../msvd/TextData/PrecomputedSentFeat/
     └── VideoSets
          └── msvdtest.txt
```

### Get data
```bash
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH; cd $ROOTPATH

# download and extract pre-trained word2vec-mini 
wget
tar zxf w2v-flickr-mini.tar.gz
# download and extract all the required data of msrvtt10k
wget
tar zxf msrvtt10k.tar.gz
# download and extract all the required data of AVS
wget
tar zxf AVS.tar.gz
# download and extract all the required data of msvd
wget
tar zxf msvd.tar.gz
# download and extract all the required data of tgif
wget
tar zxftgif.tar.gz
```


## Scripts for training, testing, and evaluation
### Training and test from sratch
```bash
# activate the conda environment
conda activate sea

# build vocabulary on the training dataset
bash do_build_vocab.sh msrvtt10ktrain
bash do_build_vocab.sh tgiftrain
bash do_build_vocab.sh msvdtrain
bash do_build_vocab.sh tgif-msrvtt10k

# choose one model config you want to ues
config=sea_resnext101-resnet152_bow_w2v
config=sea_resnext101-resnet152_bow_w2v_gru
config=sea_resnext101-resnet152_bow_w2v_bigru
config=sea_resnext101-resnet152_bow_w2v_bert
config=sea_resnext101-resnet152_bow_w2v_gru_bert
config=sea_resnext101-resnet152_bow_w2v_bigru_bert

# use the first GPU on your device
gpu_id=0 

# do train and test on msrvtt10k
bash do_train_and_test_msrvtt10k $config $gpu_id

# do train and test on msvd
bash do_train_and_test_msvd.sh $config $gpu_id

# do train and test on tgif
bash do_train_and_test_tgif.sh $config $gpu_id

# for AVS, do train on tgif-msrvtt10k and test on iacc.3 or v3c1.
bash do_train_tgif-msrvtt10k.sh $config $gpu_id
bash do_test_iacc.3.sh $config $gpu_id
bash do_test_v3c1.sh $config $gpu_id
# for AVS, do evaluation on topics of tv16-20
cd tv-avs-eval
do_eval.sh $config
```
### Test and evaluate a pre-trained model
to do

## Repoted performance

### msrvtt10k
Sentence encoder | Model | R@1 | R@5 | R@5 | Med r | mAP|
|--- | ---| ---| ---| ---| ---| ---|
|{BoW, w2v}| w2vv++| 10.9| 29.1| 39.9| 19| 20.2|
| | SEA| 11.6| 30.6| 41.6| 17| 21.3(↑5.4%) |
|{BoW, w2v, GRU}|w2vv++|11.1 |29.6 |40.5 |18 |20.6 |
||SEA|12.2 |31.9 |43.1 |15 |22.1(↑7.3%) |
|{BoW, w2v, bi-GRU}|w2vv++|11.3 | 29.9 |40.6 |18 |20.8|
||SEA|12.4 |32.1 |43.3 |15 |22.3(↑7.2%)|
|{BoW, w2v, BERT}|w2vv++|12.3 |31.8 |43.0 |15 |22.2|
||SEA|12.8 |33.1| 44.6 |14 |23.0(↑3.6%)|
|{BoW, w2v, GRU, BERT}|w2vv++|12.1 |31.7 |42.7 |16 |22.0|
||SEA|13.0 |33.6| 44.9| 14 |23.3(↑5.9%)|
|{BoW, w2v, biGRU, BERT}|w2vv++|12.0 |31.3 |42.3| 16 |21.8|
||SEA|13.1 |33.4| 45.0| 14 |23.3(↑6.9%)|



### AVS
-

### tgif
-

### msvd
-



## References

```
@ARTICLE{tmm2020-sea,
  title={SEA: Sentence Encoder Assembly for Video Retrieval by Textual Queries}, 
  author={Xirong Li and Fangming Zhou and Chaoxi Xu and Jiaqi Ji and Gang Yang},
  journal={IEEE Transactions on Multimedia}, 
  year={2020},
  doi={10.1109/TMM.2020.3042067}}
```
