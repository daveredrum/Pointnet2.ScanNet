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
from lib.config import CONF

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    scene_list = sorted(scene_list, key=lambda x: int(x.split("_")[0][5:]))

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

def filter_points(coords, preds, targets, weights):
    assert coords.shape[0] == preds.shape[0] == targets.shape[0] == weights.shape[0]
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered, target_filtered, weight_filtered = coords[coord_ids], preds[coord_ids], targets[coord_ids], weights[coord_ids]

    return coord_filtered, pred_filtered, target_filtered, weight_filtered

def compute_acc(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class = [0 for _ in range(CONF.NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]

    labelweights = np.zeros(CONF.NUM_CLASSES)
    labelweights_vox = np.zeros(CONF.NUM_CLASSES)

    correct = np.sum(preds == targets) # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += targets.shape[0]
    tmp,_ = np.histogram(targets,range(CONF.NUM_CLASSES+1))
    labelweights += tmp
    for l in seen_classes:
        total_seen_class[l] += np.sum(targets==l)
        total_correct_class[l] += np.sum((preds==l) & (targets==l))

    _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    total_correct_vox += np.sum(uvlabel[:,0]==uvlabel[:,1])
    total_seen_vox += uvlabel[:,0].shape[0]
    tmp,_ = np.histogram(uvlabel[:,0],range(CONF.NUM_CLASSES+1))
    labelweights_vox += tmp
    for l in seen_classes:
        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights.astype(np.float32)/np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = labelweights_vox
    voxcaliacc = np.average(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox,dtype=np.float)+1e-8),weights=caliweights)

    pointacc_per_class = np.zeros(CONF.NUM_CLASSES)
    voxacc_per_class = np.zeros(CONF.NUM_CLASSES)
    for l in seen_classes:
        pointacc_per_class[l] = total_correct_class[l]/(total_seen_class[l] + 1e-8)
        voxacc_per_class[l] = total_correct_class_vox[l]/(total_seen_class_vox[l] + 1e-8)

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask

def compute_miou(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    pointmiou = np.zeros(CONF.NUM_CLASSES)
    voxmiou = np.zeros(CONF.NUM_CLASSES)

    uvidx, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    for l in seen_classes:
        target_label = np.arange(targets.shape[0])[targets==l]
        pred_label = np.arange(preds.shape[0])[preds==l]
        num_intersection_label = np.intersect1d(pred_label, target_label).shape[0]
        num_union_label = np.union1d(pred_label, target_label).shape[0]
        pointmiou[l] = num_intersection_label / (num_union_label + 1e-8)

        target_label_vox = uvidx[(uvlabel[:, 0] == l)]
        pred_label_vox = uvidx[(uvlabel[:, 1] == l)]
        num_intersection_label_vox = np.intersect1d(pred_label_vox, target_label_vox).shape[0]
        num_union_label_vox = np.union1d(pred_label_vox, target_label_vox).shape[0]
        voxmiou[l] = num_intersection_label_vox / (num_union_label_vox + 1e-8)

    return pointmiou, voxmiou, mask

def eval_one_batch(args, model, data):
    # unpack
    coords, feats, targets, weights, _ = data
    coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()

    # feed
    preds = forward(args, model, coords, feats)

    # eval
    coords = coords.squeeze(0).view(-1, 3).cpu().numpy()     # (CK*N, C)
    preds = preds.squeeze(0).view(-1).cpu().numpy()          # (CK*N, C)
    targets = targets.squeeze(0).view(-1).cpu().numpy()      # (CK*N, C)
    weights = weights.squeeze(0).view(-1).cpu().numpy()      # (CK*N, C)
    pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, acc_mask = compute_acc(coords, preds, targets, weights)
    pointmiou, voxmiou, miou_mask = compute_miou(coords, preds, targets, weights)
    assert acc_mask.all() == miou_mask.all()

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, pointmiou, voxmiou, acc_mask


def eval_wholescene(args, model, dataloader):
    # init
    pointacc_list = []
    pointacc_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxacc_list = []
    voxacc_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxcaliacc_list = []
    pointmiou_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxmiou_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    masks = np.zeros((len(dataloader), CONF.NUM_CLASSES))

    # iter
    for load_idx, data in enumerate(dataloader):
        # feed
        pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, pointmiou, voxmiou, mask = eval_one_batch(args, model, data)

        # dump
        pointacc_list.append(pointacc)
        pointacc_per_class_array[load_idx] = pointacc_per_class
        voxacc_list.append(voxacc)
        voxacc_per_class_array[load_idx] = voxacc_per_class
        voxcaliacc_list.append(voxcaliacc)
        pointmiou_per_class_array[load_idx] = pointmiou
        voxmiou_per_class_array[load_idx] = voxmiou
        masks[load_idx] = mask

    return pointacc_list, pointacc_per_class_array, voxacc_list, voxacc_per_class_array, voxcaliacc_list, pointmiou_per_class_array, voxmiou_per_class_array, masks

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
    pointacc_list, pointacc_per_class_array, voxacc_list, voxacc_per_class_array, voxcaliacc_list, pointmiou_per_class_array, voxmiou_per_class_array, masks = eval_wholescene(args, model, dataloader)
    
    avg_pointacc = np.mean(pointacc_list)
    avg_pointacc_per_class = np.sum(pointacc_per_class_array * masks, axis=0)/np.sum(masks, axis=0)

    avg_voxacc = np.mean(voxacc_list)
    avg_voxacc_per_class = np.sum(voxacc_per_class_array * masks, axis=0)/np.sum(masks, axis=0)

    avg_voxcaliacc = np.mean(voxcaliacc_list)
    
    avg_pointmiou_per_class = np.sum(pointmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
    avg_pointmiou = np.mean(avg_pointmiou_per_class)

    avg_voxmiou_per_class = np.sum(voxmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
    avg_voxmiou = np.mean(avg_voxmiou_per_class)

    # report
    print()
    print("Point accuracy: {}".format(avg_pointacc))
    print("Point accuracy per class: {}".format(np.mean(avg_pointacc_per_class)))
    print("Voxel accuracy: {}".format(avg_voxacc))
    print("Voxel accuracy per class: {}".format(np.mean(avg_voxacc_per_class)))
    print("Calibrated voxel accuracy: {}".format(avg_voxcaliacc))
    print("Point miou: {}".format(avg_pointmiou))
    print("Voxel miou: {}".format(avg_voxmiou))
    print()

    print("Point acc/voxel acc/point miou/voxel miou per class:")
    for l in range(CONF.NUM_CLASSES):
        print("Class {}: {}/{}/{}/{}".format(CONF.NYUCLASSES[l], avg_pointacc_per_class[l], avg_voxacc_per_class[l], avg_pointmiou_per_class[l], avg_voxmiou_per_class[l]))


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