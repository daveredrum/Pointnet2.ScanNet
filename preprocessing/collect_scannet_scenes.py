import os
import sys
import json
import time
import numpy as np

sys.path.append(".")
from scannet_util import g_label_names, g_raw2scannet
from lib.pc_util import read_ply_xyzrgb
from lib.utils import get_eta
from lib.config import CONF

CLASS_NAMES = g_label_names
RAW2SCANNET = g_raw2scannet

def collect_one_scene_data_label(scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(CONF.SCANNET_DIR, scene_name)
    mesh_seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    #print mesh_seg_filename
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        #print len(seg)
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    
    # Raw points in XYZRGBA
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgb(ply_filename)
    
    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    # annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
    annotation_filename = os.path.join(data_folder, '%s_vh_clean.aggregation.json'%(scene_name))
    #print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])
    
    #print len(instance_segids)
    #print labels
    
    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids),:]
        instance_points_list.append(instance_points)
        instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)   
        label = RAW2SCANNET[labels[i]]
        label = CLASS_NAMES.index(label)
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)
       
    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:,0:6] # XYZRGB, disregarding the A
    instance_labels = np.concatenate(instance_labels_list, 0) 
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)
    np.save(out_filename, data)

if __name__=='__main__':
    os.makedirs(CONF.PREP_SCANS, exist_ok=True)
    
    for i, scene_name in enumerate(CONF.SCENE_NAMES):
        try:
            start = time.time()
            out_filename = scene_name+'.npy' # scene0000_00.npy
            collect_one_scene_data_label(scene_name, os.path.join(CONF.PREP_SCANS, out_filename))
            
            # report
            num_left = len(CONF.SCENE_NAMES) - i - 1
            eta = get_eta(start, time.time(), 0, num_left)
            print("preprocessed {}, {} left, ETA: {}h {}m {}s".format(
                scene_name,
                num_left,
                eta["h"],
                eta["m"],
                eta["s"]
            ))

        except Exception as e:
            print(scene_name+'ERROR!!')
            print(e)

    print("done!")