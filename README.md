# Pointnet2.ScanNet
PointNet++ Semantic Segmentation on ScanNet with CUDA acceleration

## Installation
### Requirements
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.0

### Install 
Install this library by running the following command:

```shell
cd pointnet2
python setup.py install
```

## Usage
### preprocess ScanNet scenes
```shell
python preprocessing/collect_scannet_scenes.py
```

### train
```shell
python train.py --batch_size 32 --epoch 500 --lr 1e-3
```

### eval
```shell
python eval.py --batch_size 32 --folder 2019-07-12_11-19-46
```
