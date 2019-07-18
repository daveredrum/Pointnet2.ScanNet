import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from plyfile import PlyElement, PlyData
from PointNet2ScanNetDataset import ScannetDatasetWholeScene, collate_wholescene

# for PointNet2.PyTorch module
import sys
sys.path.append(".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pointnet2/'))
from lib.config import CONF

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]

def forward(args, model, coords, feats):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), args.batch_size, 0), torch.split(feats.squeeze(0), args.batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0) # (CK, N, C)
    outputs = pred.max(2)[1]

    return outputs

def filter_points(coords, preds):
    assert coords.shape[0] == preds.shape[0]
    palette = create_color_palette()

    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered = coords[coord_ids], preds[coord_ids]
    filtered = []
    for point_idx in range(coord_filtered.shape[0]):
        filtered.append(
            [
                coord_filtered[point_idx][0],
                coord_filtered[point_idx][1],
                coord_filtered[point_idx][2],
                palette[pred_filtered[point_idx]][0],
                palette[pred_filtered[point_idx]][1],
                palette[pred_filtered[point_idx]][2]
            ]
        )
    
    return np.array(filtered)


def predict_label(args, model, dataloader):
    output_coords, output_preds = [], []
    print("predicting labels...")
    for data in dataloader:
        # unpack
        coords, feats, targets, weights, _ = data
        coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()

        # feed
        preds = forward(args, model, coords, feats)

        # dump
        coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
        preds = preds.view(-1).cpu().numpy()
        output_coords.append(coords)
        output_preds.append(preds)

    print("filtering points...")
    output_coords = np.concatenate(output_coords, axis=0)
    output_preds = np.concatenate(output_preds, axis=0)
    filtered = filter_points(output_coords, output_preds)

    return filtered

def visualize(args, preds):
    vertex = []
    for i in range(preds.shape[0]):
        vertex.append(
            (
                preds[i][0],
                preds[i][1],
                preds[i][2],
                preds[i][3],
                preds[i][4],
                preds[i][5],
            )
        )

    vertex = np.array(
        vertex,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    output_pc = PlyElement.describe(vertex, "vertex")
    output_pc = PlyData([output_pc])
    output_root = os.path.join(CONF.OUTPUT_ROOT, args.folder, "preds")
    os.makedirs(output_root, exist_ok=True)
    output_pc.write(os.path.join(output_root, "{}.ply".format(args.scene_id)))


def get_scene_list(args):
    scene_list = []
    if args.scene_id:
        scene_list.append(args.scene_id)
    else:
        raise ValueError("Select a scene to visualize")

    return scene_list

def evaluate(args):
    # prepare data
    print("preparing data...")
    scene_list = get_scene_list(args)
    dataset = ScannetDatasetWholeScene(scene_list, is_weighting=True)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)

    # load model
    print("loading model...")
    model_path = os.path.join(CONF.OUTPUT_ROOT, args.folder, "model.pth")
    Pointnet = importlib.import_module("pointnet2_msg_semseg")
    model = Pointnet.get_model(num_classes=21).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # predict
    print("predicting...")
    preds = predict_label(args, model, dataloader)

    # visualize
    print("visualizing...")
    visualize(args, preds)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='output folder containing the best model from training', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch/chunk', default=8)
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument("--scene_id", type=str, default=None)
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    evaluate(args)
    print("done!")