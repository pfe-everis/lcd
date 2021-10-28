import json

import numpy as np
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
    # Wrtiting target PCD to disk, so it can be used later.
    with open("/tmp/target.pcd", "w") as f:
        f.write(target.file.read().decode("utf-8"))
        target.file.close()

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

    success = True
    if result.transformation.trace() == 4.0:
        success = False

    information = open3d.registration.get_information_matrix_from_point_clouds(
        source_points, target_points, threshold, result.transformation
    )
    n = min(len(source_points.points), len(target_points.points))
    if (information[5, 5] / n) < 0.3:  # overlap threshold
        success = False

    if not success:
        return {
            "success": False,
            "message": "Could not align given point clouds.",
            "data": None,
        }
    else:
        mat_str = np.array2string(result.transformation, separator=",")
        print(f"[INFO] Alignment successful. {mat_str}")
        return {
            "success": True,
            "message": "Successfully aligned the given point clouds.",
            "data": mat_str,
        }
