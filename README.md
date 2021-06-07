# SEA for Cross-Modal Video Retrieval
Source code of our TMM paper: [SEA: Sentence Encoder Assembly for Video Retrieval by Textual Queries](https://doi.org/10.1109/TMM.2020.3042067). 
The code assumes [video-level CNN features](https://github.com/xuchaoxi/video-cnn-feat) have been extracted. 


## Environment
* Ubuntu 16.04
* cuda 10.1
* python 3.7
* PyTorch 1.4.0
* tensorboard 2.1.0
* numpy 1.19.5

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.
```
conda create -n sea python==3.7
conda activate sea
git clone https://github.com/li-xirong/sea.git
cd sea
pip install -r requirements.txt
```

## Data


### Get data
```bash
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH; cd $ROOTPATH

# download a mini-version of a word2vec model trained on Flickr tags. 
wget http://lixirong.net/data/sea/w2v-flickr-mini.tar.gz
tar xzf w2v-flickr-mini.tar.gz

# download the MSVD data package 
wget http://lixirong.net/data/sea/msvd.tar.gz
tar xzf msvd.tar.gz
```

### Data organization

We use MSVD as an example. Other collections such as MSR-VTT and TGIF are organized in a similar style.

```bash
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
│   │   └── PrecomputedSentFeat -> ../../msvd/TextData/PrecomputedSentFeat/
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
     │   └── PrecomputedSentFeat -> ../../msvd/TextData/PrecomputedSentFeat/
     └── VideoSets
          └── msvdtest.txt
```



## Tutorial scripts

### Train and test from sratch
```bash
# activate the conda environment
conda activate sea

# choose a specific text-encoder configuraiton
config=sea_resnext101-resnet152_bow_w2v_bert

# choose a specifc GPU card
gpu_id=0 

# do train and test on msvd
bash do_train_and_test_msvd.sh $config $gpu_id
```

### Test and evaluate a pre-trained model
to do

## Performance

### msrvtt10k
Sentence encoder        |Model |R@1 |R@5 |R@10|Med r|mAP        |
|---                    |---   |--- |--- |--- |---  |---        |
|{BoW, w2v}             |w2vv++|10.9|29.1|39.9|19   |20.2       |
|                       |SEA   |11.6|30.6|41.6|17   |21.3(↑5.4%)|
|{BoW, w2v, GRU}        |w2vv++|11.1|29.6|40.5|18   |20.6       |
|                       |SEA   |12.2|31.9|43.1|15   |22.1(↑7.3%)|
|{BoW, w2v, bi-GRU}     |w2vv++|11.3|29.9|40.6|18   |20.8       |
|                       |SEA   |12.4|32.1|43.3|15   |22.3(↑7.2%)|
|{BoW, w2v, BERT}       |w2vv++|12.3|31.8|43.0|15   |22.2       |
|                       |SEA   |12.8|33.1|44.6|14   |23.0(↑3.6%)|
|{BoW, w2v, GRU, BERT}  |w2vv++|12.1|31.7|42.7|16   |22.0       |
|                       |SEA   |13.0|33.6|44.9|14   |23.3(↑5.9%)|
|{BoW, w2v, biGRU, BERT}|w2vv++|12.0|31.3|42.3|16   |21.8       |
|                       |SEA   |13.1|33.4|45.0|14   |23.3(↑6.9%)|

### AVS
Sentence encoder        |Model |TV16|TV17|TV18|TV19|MEAN         |    
|---                    |---   |--- |--- |--- |--- |---          |
|{BoW, w2v}             |w2vv++|14.4|21.8|11.1|14.3|15.4         |
|                       |SEA   |15.7|23.4|12.8|16.6|17.1("11.2%) |
|{BoW, w2v, GRU}        |w2vv++|16.2|22.3|10.1|13.9|15.6         |   
|                       |SEA   |15.0|23.4|12.2|16.6|16.8(↑7.5%)  |
|{BoW, w2v, bi-GRU}     |w2vv++|16.1|21.7|10.4|13.5|15.4         |
|                       |SEA   |16.4|22.8|12.5|16.7|17.1(↑10.9%) |
|{BoW, w2v, BERT}       |w2vv++|15.1|22.5|10.2|12.8|15.2         |
|                       |SEA   |15.3|22.8|12.1|14.8|16.3(↑7.3%)  |
|{BoW, w2v, GRU, BERT}  |w2vv++|14.3|19.3|9.3 |10.1|13.3         |
|                       |SEA   |16.0|23.1|12.1|15.4|16.7(↑25.7%) |
|{BoW, w2v, biGRU, BERT}|w2vv++|15.8|20.6|9.0 |10.5|14.0         |
|                       |SEA   |15.9|22.9|11.7|15.5|16.5(↑18.1%) |

### tgif
Model                        |R@1 |R@5 |R@10|Med r|mAP | 
|---                         |--- |--- |--- |---  |--- |
|SEA({BoW, w2v, GRU})        |10.2|23.6|31.3|41   |17.2|
|SEA({BoW, w2v, BERT})       |10.7|24.4|31.9|37   |17.9|
|SEA({BoW, w2v, GRU, BERT})  |11.1|25.2|32.7|36   |18.4|
|SEA({BoW, w2v, biGRU, BERT})|11.1|25.2|32.8|35   |18.5|


### msvd
Model                        |R@1 |R@5 |R@10|Med r|mAP | 
|---                         |--- |--- |--- |---  |--- |
|SEA({BoW, w2v, GRU})        |23.2|52.9|66.2|5    |37.2|
|SEA({BoW, w2v, BERT})       |24.6|55.0|67.9|4    |38.7|
|SEA({BoW, w2v, GRU, BERT})  |24.4|54.1|67.6|5    |38.3|
|SEA({BoW, w2v, biGRU, BERT})|23.9|53.9|67.3|5    |38.0|




## References

```
@ARTICLE{tmm2020-sea,
  title={SEA: Sentence Encoder Assembly for Video Retrieval by Textual Queries}, 
  author={Xirong Li and Fangming Zhou and Chaoxi Xu and Jiaqi Ji and Gang Yang},
  journal={IEEE Transactions on Multimedia}, 
  year={2020},
  doi={10.1109/TMM.2020.3042067}}
```
