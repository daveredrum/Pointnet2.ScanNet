# Pointnet2.ScanNet
PointNet++ Semantic Segmentation on ScanNet in PyTorch with CUDA acceleration based on the original [PointNet++ repo](https://github.com/charlesq34/pointnet2) and [the PyTorch implementation with CUDA](https://github.com/sshaoshuai/Pointnet2.PyTorch)

## Performance
The semantic segmentation results in percentage on the ScanNet train/val split in `data/`.
<table>
  <tr>
    <td>Avg</td><td>Floor</td><td>Wall</td><td>Cabinet</td><td>Bed</td><td>Chair</td><td>Sofa</td><td>Table</td><td>Door</td><td>Window</td><td>Bookshelf</td><td>Picture</td><td>Counter</td><td>Desk</td><td>Curtain</td><td>Refrigerator</td><td>Bathtub</td><td>Shower</td><td>Toilet</td><td>Sink</td><td>Others</td>
  </tr>
  <tr>
    <td><b>50.62</b></td><td>90.96</td><td>63.87</td><td>35.21</td><td>56.75</td><td>62.43</td><td>68.46</td><td>47.15</td><td>36.12</td><td>34.12</td><td>25.62</td><td>23.58</td><td>41.46</td><td>42.73</td><td>32.38</td><td>44.12</td><td>64.93</td><td>63.90</td><td>74.04</td><td>58.13</td><td>46.40</td>
  </tr>
</table>

The pretrained models: [SSG](https://www.dropbox.com/s/wunli6uxqf2llor/pointnet2_semseg_ssg_xyzrgb.pth?dl=0) [MSG](https://www.dropbox.com/s/3cokg7ediutei1d/pointnet2_semseg_msg_xyzrgb.pth?dl=0) 

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

### Prepare multiview features (optional)
1. Download the ScanNet frames [here](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip) (~13GB) and unzip it.

2. Extract the multiview features from ENet:
```shell
python compute_multiview_features.py
```

3. Generate the projection mapping between image and point cloud
```shell
python compute_multiview_projection.py
```

4. Project the multiview features from image space to point cloud
```shell
python project_multiview_features.py
```

## Usage
### preprocess ScanNet scenes
Parse the ScanNet data into `*.npy` files and save them in `preprocessing/scannet_scenes/`
```shell
python preprocessing/collect_scannet_scenes.py
```
#### sanity check
Don't forget to visualize the preprocessed scenes to check the consistency
```shell
python preprocessing/visualize_prep_scene.py --scene_id <scene_id>
```
The visualized `<scene_id>.ply` is stored in `preprocessing/label_point_clouds/`

### train
Train the PointNet++ semantic segmentation model on ScanNet scenes
```shell
python train.py
```
The trained models and logs will be saved in `outputs/<time_stamp>/`
> Note: please refer to [train.py](https://github.com/daveredrum/Pointnet2.ScanNet/blob/master/train.py) for more training settings

### eval
Evaluate the trained models and report the segmentation performance in point accuracy, voxel accuracy and calibrated voxel accuracy
```shell
python eval.py --folder <time_stamp>
```

### vis
Visualize the semantic segmentation results on points in a given scene
```shell
python visualize.py --folder <time_stamp> --scene_id <scene_id>
```
The generated `<scene_id>.ply` is stored in `outputs/<time_stamp>/preds`. See the class palette [here](http://kaldir.vc.in.tum.de/scannet_benchmark/img/legend.jpg)

## Acknowledgement
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo.
* [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): Initial work of PyTorch implementation of PointNet++ with CUDA acceleration.
