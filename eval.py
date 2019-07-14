import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from PointNet2ScanNetDataset import ScannetDatasetWholeScene, collate_wholescene
from pc_util import point_cloud_label_to_surface_voxel_label_fast

# for PointNet2.PyTorch module
import sys
sys.path.append(".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointnet2/'))

# META
NUM_CLASSES = 21

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    return scene_list

def forward(args, model, coords, feats):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), args.batch_size, 0), torch.split(feats.squeeze(0), args.batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0).unsqueeze(0) # (1, CK, N, C)
    outputs = pred.max(3)[1]

    return outputs

def compute_acc(coords, preds, targets, weights):
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)

    correct = np.sum((preds == targets) & (targets>0) & (weights>0)) # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += np.sum((targets>0) & (weights>0))
    tmp,_ = np.histogram(targets,range(22))
    labelweights += tmp
    for l in range(NUM_CLASSES):
        total_seen_class[l] += np.sum((targets==l) & (weights>0))
        total_correct_class[l] += np.sum((preds==l) & (targets==l) & (weights>0))

    for b in range(coords.shape[0]):
        _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords[b,weights[b,:]>0,:], np.concatenate((np.expand_dims(targets[b,weights[b,:]>0],1),np.expand_dims(preds[b,weights[b,:]>0],1)),axis=1), res=0.02)
        total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
        total_seen_vox += np.sum(uvlabel[:,0]>0)
        tmp,_ = np.histogram(uvlabel[:,0],range(22))
        labelweights_vox += tmp
        for l in range(NUM_CLASSES):
            total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
            total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    voxcaliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)

    return pointacc, voxacc, voxcaliacc

def eval_one_batch(args, model, data):
    # unpack
    coords, feats, targets, weights, _ = data
    coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()

    # feed
    preds = forward(args, model, coords, feats)

    # eval
    coords = coords.squeeze(0).cpu().numpy()     # (CK, N, C)
    preds = preds.squeeze(0).cpu().numpy()       # (CK, N, C)
    targets = targets.squeeze(0).cpu().numpy()   # (CK, N, C)
    weights = weights.squeeze(0).cpu().numpy()   # (CK, N, C)
    pointacc, voxacc, voxcaliacc = compute_acc(coords, preds, targets, weights)

    return pointacc, voxacc, voxcaliacc


def eval_wholescene(args, model, dataloader):
    # init
    pointacc_list = []
    voxacc_list = []
    voxcaliacc_list = []

    # iter
    for data in dataloader:
        # feed
        pointacc, voxacc, voxcaliacc = eval_one_batch(args, model, data)

        # dump
        pointacc_list.append(pointacc)
        voxacc_list.append(voxacc)
        voxcaliacc_list.append(voxcaliacc)

    return pointacc_list, voxacc_list, voxcaliacc_list

def evaluate(args):
    # prepare data
    print("preparing data...")
    scene_list = get_scene_list("python/Mesh2Loc/data/scannetv2_val.txt")
    dataset = ScannetDatasetWholeScene(scene_list, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)

    # load model
    print("loading model...")
    model_path = os.path.join(CONF.OUTPUT_ROOT, args.folder, "model.pth")
    Pointnet = importlib.import_module("pointnet2_msg_semseg")
    model = Pointnet.get_model(num_classes=21).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # eval
    print("evaluating...")
    pointacc_list, voxacc_list, voxcaliacc_list = eval_wholescene(args, model, dataloader)
    avg_pointacc = np.mean(pointacc_list)
    avg_voxacc = np.mean(voxacc_list)
    avg_voxcaliacc = np.mean(voxcaliacc_list)

    # report
    print()
    print("Point accuracy: {}".format(avg_pointacc))
    print("Voxel-based point accuracy: {}".format(avg_voxacc))
    print("Calibrated point accuracy: {}".format(avg_voxcaliacc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='output folder containing the best model from training', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch/chunk', default=8)
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    evaluate(args)