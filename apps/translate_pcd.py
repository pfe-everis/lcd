import argparse

import numpy as np
import open3d
from lcd.models import *

parser = argparse.ArgumentParser()
parser.add_argument("source", help="path to the source point cloud")
parser.add_argument("target", help="path to the target point cloud")
args = parser.parse_args()

source = open3d.io.read_point_cloud(args.source)
target = open3d.io.read_point_cloud(args.target)

print("Visualizing PCDs...")
source.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
target.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
source.paint_uniform_color([1, 0.706, 0])
target.paint_uniform_color([0, 0.651, 0.929])

deg = 0
deg = np.deg2rad(deg)
# x = np.array(
#     [
#         [np.cos(deg), 0, -np.sin(deg), 10],
#         [0, 1, 0, 0],
#         [np.sin(deg), 0, np.cos(deg), 0],
#         [0, 0, 0, 1],
#     ]
# )
x = [
    [
        0.9965021308818449,
        -0.00424166870863045,
        0.08345963931451571,
        -0.5584476831680415,
    ],
    [
        0.006975143829066725,
        0.9994477929590735,
        -0.032487790288069585,
        0.06916415139061667,
    ],
    [
        -0.08327574987057548,
        0.032956295237844685,
        0.9959814416381914,
        1.5672740851174451,
    ],
    [0.0, 0.0, 0.0, 1.0],
]

source.transform(x)
open3d.visualization.draw_geometries([source, target])

# open3d.io.write_point_cloud("target-trans-10x.pcd", source)
