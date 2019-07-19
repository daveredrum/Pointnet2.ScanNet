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

### Configure
Change the path configurations for the ScanNet data in `lib/config.py`

## Usage
### preprocess ScanNet scenes
Parse the ScanNet data into `*.npy` files and save them in `preprocessing/scannet_scenes/`
```shell
python preprocessing/collect_scannet_scenes.py
```
#### Sanity check
Don't forget to visualize the preprocessed scenes to check the consistency
```shell
python preprocessing/visualize_prep_scene.py --scene_id <scene_id>
```
The visualized `<scene_id>.ply` is stored in `preprocessing/label_point_clouds/`

### train
Train the PointNet++ semantic segmentation model on ScanNet scenes
```shell
python train.py --batch_size 32 --epoch 500 --lr 1e-3 --verbose 10 --weighting
```
The trained models and logs will be saved in `outputs/<time_stamp>/`
> Note: please refer to [train.py](https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/train.py) for more training settings

### eval
Evaluate the trained models and report the segmentation performance in point accuracy, voxel accuracy and calibrated voxel accuracy
```shell
python eval.py --batch_size 32 --folder <time_stamp>
```

### Vis
Visualize the semantic segmentation results on points in a given scene
```shell
python visualize.py --batch_size 32 --folder <time_stamp> --scene_id <scene_id>
```
The generated `<scene_id>.ply` is stored in `outputs/<time_stamp>/preds`

## Acknowledgement
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo.
* [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): Initial work of PyTorch implementation of PointNet++ with CUDA acceleration.
