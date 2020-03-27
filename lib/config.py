import os
from easydict import EasyDict

CONF = EasyDict()

# BASE PATH
CONF.ROOT = "/home/davech2y/Pointnet2.ScanNet" # TODO change this
CONF.SCANNET_DIR =  "/mnt/canis/Datasets/ScanNet/public/v2/scans" # TODO change this
CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))

CONF.PREP = os.path.join(CONF.ROOT, "preprocessing")
CONF.PREP_SCANS = os.path.join(CONF.PREP, "scannet_scenes")
CONF.SCAN_LABELS = os.path.join(CONF.PREP, "label_point_clouds")
CONF.OUTPUT_ROOT = os.path.join(CONF.ROOT, "outputs")

CONF.SCANNETV2_TRAIN = os.path.join(CONF.ROOT, "data/scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.ROOT, "data/scannetv2_val.txt")
CONF.SCANNETV2_FILE = os.path.join(CONF.PREP_SCANS, "{}.npy") # scene_id
CONF.SCANNETV2_LABEL = os.path.join(CONF.SCAN_LABELS, "{}.ply") # scene_id

CONF.NYUCLASSES = [
    'floor', 
    'wall', 
    'cabinet', 
    'bed', 
    'chair', 
    'sofa', 
    'table', 
    'door', 
    'window', 
    'bookshelf', 
    'picture', 
    'counter', 
    'desk', 
    'curtain', 
    'refrigerator', 
    'bathtub', 
    'shower curtain', 
    'toilet', 
    'sink', 
    'otherprop'
]
CONF.NUM_CLASSES = len(CONF.NYUCLASSES)
CONF.PALETTE = [
    (152, 223, 138),		# floor
    (174, 199, 232),		# wall
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
    (247, 182, 210),		# desk
    (219, 219, 141),		# curtain
    (255, 127, 14), 		# refrigerator
    (227, 119, 194),		# bathtub
    (158, 218, 229),		# shower curtain
    (44, 160, 44),  		# toilet
    (112, 128, 144),		# sink
    (82, 84, 163),          # otherfurn
]