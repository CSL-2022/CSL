# Cascade Shard Learning for Click-Through Rate Prediction

## Prerequisites
- Python 2.x
- Tensorflow 1.15.0

## Data
- [Taobao Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1)
- [Amazon Data](http://jmcauley.ucsd.edu/data/amazon/)

### Taobao Prepare
First download [Taobao Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1) 
to get "UserBehavior.csv.zip", then execute the following command.
```
sh prepare_taobao.sh
```

## Running
```
python script/train_taobao.py -p train --random_seed 3 --model_type DNN --learn_type FCN
python script/train_taobao.py -p test --random_seed 3 --model_type DNN --learn_type FCN
```
The model_type below had been supported: 
- DNN
- DIN
- DIEN
- SIM

The learn_type below had been supported: 
- FCN
- SL
- CSL

