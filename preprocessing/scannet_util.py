import os
import sys

sys.path.append(".")
from lib.config import CONF

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']

def get_raw2scannet_label_map():
    # lines = [line.rstrip() for line in open('scannet-labels.combined.tsv')]
    lines = [line.rstrip() for line in open(os.path.join(CONF.PREP, 'scannetv2-labels.combined.tsv'))]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


g_raw2scannet = get_raw2scannet_label_map()
