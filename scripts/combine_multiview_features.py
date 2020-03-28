import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF

ENET_FEATURE_DATABASE = CONF.MULTIVIEW
SCANNET_DATA = CONF.PREP_SCANS

def get_scene_list():
    with open(SCANNET_LIST, 'r') as f:
        scene_list = sorted(list(set(f.read().splitlines())))
    
    return scene_list

def load_scene(scene_list):
    scene_data = {}
    for scene_id in scene_list:
        scene_data[scene_id] = np.load(os.path.join(SCANNET_DATA, scene_id)+".npy")[:, :3]

    return scene_data

if __name__ == "__main__":
    scene_list = get_scene_list()
    scene_data = load_scene()
    multiview_data = h5py.File(ENET_FEATURE_DATABASE, "w", libver="latest")

    print("combining features to point cloud")
    for scene_id in scene_list:
        multiview_features = multiview_data.get(scene_id)[()]
        scene_data[scene_id] = np.concatenate((scene_data[scene_id], multiview_features), 1)

        np.save(scene_data[scene_id], os.path.join(SCANNET_DATA, scene_id)+".npy")

    print("done!")