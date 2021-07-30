import os
import argparse
import numpy as np
from plyfile import PlyElement, PlyData

import sys
sys.path.append(".")
from lib.config import CONF

def visualize(args):
    print("visualizing...")
    scene = np.load(CONF.SCANNETV2_FILE.format(args.scene_id))

    vertex = []
    for i in range(scene.shape[0]):
        vertex.append(
            (
                scene[i][0],
                scene[i][1],
                scene[i][2],
                CONF.PALETTE[int(scene[i][-1])][0],
                CONF.PALETTE[int(scene[i][-1])][1],
                CONF.PALETTE[int(scene[i][-1])][2]
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
    os.makedirs(CONF.SCAN_LABELS, exist_ok=True)
    output_pc.write(CONF.SCANNETV2_LABEL.format(args.scene_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, required=True)
    args = parser.parse_args()

    visualize(args)
    print("done!")