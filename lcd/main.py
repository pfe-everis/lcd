import json

import numpy as np
from numpy.core.fromnumeric import size
import open3d
import torch
from fastapi import FastAPI
from fastapi.datastructures import UploadFile
from fastapi.params import File

from models import PointNetAutoencoder
from utils import compute_lcd_descriptors, download_file, extract_uniform_patches

app = FastAPI()


@app.post("/")
async def match_pcds(source: str, target: UploadFile = File(...)):
    # deg = 90
    # deg = np.deg2rad(deg)
    # x = np.array(
    #     [
    #         [np.cos(deg), 0, -np.sin(deg), 0],
    #         [0, 1, 0, 0],
    #         [np.sin(deg), 0, np.cos(deg), 0],
    #         [0, 0, 0, 1],
    #     ]
    # )

    # row_1 = list(x[0])
    # row_2 = list(x[1])
    # row_3 = list(x[2])
    # row_4 = list(x[3])
    # res = {
    #     "success": True,
    #     "message": "Successfully aligned the given point clouds.",
    #     "row1": row_1,
    #     "row2": row_2,
    #     "row3": row_3,
    #     "row4": row_4,
    # }
    # print(f"[INFO] Return body: {res}")
    # return res
    print("[INFO] POST received, starting to match...")

    # Wrtiting target PCD to disk, so it can be used later.
    with open("/tmp/target.pcd", "w") as f:
        content = target.file.read().decode("utf-8")
        f.write(content)
        target.file.close()
        if len(content.split("\n")) <= 12:
            print("[ERROR] Target file is empty.")
            return {"success": False, "message": "Target file is empty."}

    print(f"[INFO] Downloading source PCD from {source}...")
    source_pcd_path: str = ""
    source_pcd_path, _ = download_file(source, "/tmp/source.pcd")
    print(f"[INFO] Done. File saved to: {source_pcd_path}")

    config = "../logs/LCD-D256/config.json"
    model_file = "../logs/LCD-D256/model.pth"
    voxel_size = 0.1
    radius = 0.15
    num_points = 1024

    config = json.load(open(config))
    device = config["device"]

    print(f"[INFO] Loading ML model...")
    model = PointNetAutoencoder(
        config["embedding_size"],
        config["input_channels"],
        config["output_channels"],
        config["normalize"],
    )
    model.load_state_dict(torch.load(model_file)["pointnet"])
    model.to(device)
    model.eval()
    print(f"[INFO] Done.")

    print(f"[INFO] Loading source PCD and extracting features...")
    source = open3d.io.read_point_cloud(source_pcd_path)

    source_points, source_patches = extract_uniform_patches(
        source, voxel_size, radius, num_points
    )
    source_descriptors = compute_lcd_descriptors(
        source_patches, model, batch_size=128, device=device
    )
    source_features = open3d.registration.Feature()
    source_features.data = np.transpose(source_descriptors)
    print(f"[INFO] Extracted {len(source_descriptors)} features from source")

    print(f"[INFO] Loading target PCD and extracting features...")
    target = open3d.io.read_point_cloud("/tmp/target.pcd")

    target_points, target_patches = extract_uniform_patches(
        target, voxel_size, radius, num_points
    )
    target_descriptors = compute_lcd_descriptors(
        target_patches, model, batch_size=128, device=device
    )
    target_features = open3d.registration.Feature()
    target_features.data = np.transpose(target_descriptors)
    print(f"[INFO] Extracted {len(target_descriptors)} features from target")

    threshold = 0.075
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        source_points,
        target_points,
        source_features,
        target_features,
        threshold,
        open3d.registration.TransformationEstimationPointToPoint(False),
        4,
        [open3d.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
        open3d.registration.RANSACConvergenceCriteria(4000000, 500),
    )

    error = ""
    success = True
    if result.transformation.trace() == 4.0:
        error = "trace"
        success = False

    information = open3d.registration.get_information_matrix_from_point_clouds(
        source_points, target_points, threshold, result.transformation
    )
    n = min(len(source_points.points), len(target_points.points))
    if (information[5, 5] / n) < 0.3:  # overlap threshold
        error = "overlap"
        success = False

    if not success:
        print(f"[INFO] Alignment failed.")
        body = {
            "success": False,
            "message": "Could not align given point clouds " + error,
        }
    else:
        row_1 = list(result.transformation[0])
        row_2 = list(result.transformation[1])
        row_3 = list(result.transformation[2])
        row_4 = list(result.transformation[3])
        print(f"[INFO] Alignment successful.")
        body = {
            "success": True,
            "message": "Successfully aligned the given point clouds.",
            "row1": row_1,
            "row2": row_2,
            "row3": row_3,
            "row4": row_4,
        }

    print(f"[INFO] Return body: {body}")
    return body
