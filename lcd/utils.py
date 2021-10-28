from http.client import HTTPMessage
from typing import Tuple

import numpy as np
import open3d
import torch
from fastapi.params import Path


def precision_recall_curve(pred, truth):
    indices = np.argsort(pred, kind="mergesort")
    pred = pred[indices]
    truth = truth[indices]
    indices = np.diff(pred).nonzero()[0]
    indices = np.r_[indices, truth.size - 1]
    tp = np.cumsum(truth)[indices]
    fp = np.cumsum(1 - truth)[indices]
    p = tp / (tp + fp)
    p[np.isnan(p)] = 0
    r = tp / tp[-1]
    fpr = fp / fp[-1]
    return p, r


def extract_uniform_patches(pcd, voxel_size, radius, num_points):
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    downsampled = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled.points)
    patches = []
    for i in range(points.shape[0]):
        k, index, _ = kdtree.search_hybrid_vector_3d(points[i], radius, num_points)
        if k < num_points:
            index = np.random.choice(index, num_points, replace=True)
        xyz = np.asarray(pcd.points)[index]
        rgb = np.asarray(pcd.colors)[index]
        xyz = (xyz - points[i]) / radius  # normalize to local coordinates
        patch = np.concatenate([xyz, rgb], axis=1)
        patches.append(patch)
    patches = np.stack(patches, axis=0)
    return downsampled, patches


def compute_lcd_descriptors(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)


def download_file(url, file_name) -> Tuple[str, HTTPMessage]:
    """
    Download a file from a url and save it to a local file.
    """
    import urllib.request

    return urllib.request.urlretrieve(url, file_name)
