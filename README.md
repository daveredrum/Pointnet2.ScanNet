# Pointnet2.ScanNet
PointNet++ Semantic Segmentation on ScanNet in PyTorch with CUDA acceleration based on the original [PointNet++ repo](https://github.com/charlesq34/pointnet2) and [the PyTorch implementation with CUDA](https://github.com/sshaoshuai/Pointnet2.PyTorch)

## Installation
### Requirements
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.0
* TensorBoardX

### Install 
Install this library by running the following command:

```shell
cd pointnet2
python setup.py install
```

### Setup
Change the path configurations for the ScanNet data in `lib/config.py`

## Usage
### preprocess ScanNet scenes
Parse the ScanNet data into `*.npy` files and save them in `preprocessing/scannet_scenes/`
```shell
python preprocessing/collect_scannet_scenes.py
```

### train
```shell
python train.py --batch_size 32 --epoch 500 --lr 1e-3
```
The trained models and logs will be saved in `outputs/<time_stamp>/`

### eval
Evaluate the trained models and report the segmentation performance in point accuracy, voxel accuracy and calibrated voxel accuracy
```shell
python eval.py --batch_size 32 --folder <time_stamp>
```

## Acknowledgement
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo.
* [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): Initial work of PyTorch implementation of PointNet++ with CUDA acceleration.
