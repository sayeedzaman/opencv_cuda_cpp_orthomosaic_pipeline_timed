#!/usr/bin/env python3

# ============================================================
# MEMORY-SAFE REMAP-GRID GENERATOR
# ============================================================
#
# This script stays in Python because it is photogrammetry projection logic:
# DSM world coordinate + COLMAP camera pose -> source image pixel.
#
# Output:
#   xmap.npy
#   ymap.npy
#   per-tile JSON metadata
#
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def read_cameras_txt(path):
    cameras = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }

    return cameras


def read_images_txt(path):
    images = {}

    with open(path, "r") as f:
        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()

        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        camera_id = int(parts[8])
        image_name = parts[9]

        R = qvec_to_rotmat(qvec)

        images[image_name] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "R": R,
            "camera_id": camera_id
        }

    return images


def project_points_to_image(points_world, image_rec, camera_rec):
    R = image_rec["R"]
    t = image_rec["tvec"].reshape(3, 1)

    pc = (R @ points_world.T + t).T

    X = pc[:, 0]
    Y = pc[:, 1]
    Z = pc[:, 2]

    valid = Z > 0

    x = X / Z
    y = Y / Z

    model = camera_rec["model"]
    params = camera_rec["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        u = f * x + cx
        v = f * y + cy

    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        u = fx * x + cx
        v = fy * y + cy

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        r2 = x*x + y*y
        radial = 1.0 + k * r2
        u = f * radial * x + cx
        v = f * radial * y + cy

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2
        u = f * radial * x + cx
        v = f * radial * y + cy

    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params

        r2 = x*x + y*y
        radial = 1.0 + k1*r2 + k2*r2*r2

        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y

        u = fx * x_dist + cx
        v = fy * y_dist + cy

    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params

        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2

        radial_num = 1.0 + k1*r2 + k2*r4 + k3*r6
        radial_den = 1.0 + k4*r2 + k5*r4 + k6*r6
        radial = radial_num / radial_den

        x_dist = x * radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * radial + p1*(r2 + 2*y*y) + 2*p2*x*y

        u = fx * x_dist + cx
        v = fy * y_dist + cy

    else:
        raise NotImplementedError(
            f"Camera model not supported yet: {model}. "
            "Add projection formula for this COLMAP camera model."
        )

    valid &= u >= 0
    valid &= v >= 0
    valid &= u < camera_rec["width"] - 1
    valid &= v < camera_rec["height"] - 1

    return u.astype(np.float32), v.astype(np.float32), valid


def get_dsm_info(dsm_path):
    ds = gdal.Open(str(dsm_path))

    if ds is None:
        raise RuntimeError(f"Cannot open DSM: {dsm_path}")

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    if nodata is None:
        nodata = -9999

    gt = ds.GetGeoTransform()

    return ds, band, gt, nodata, ds.RasterXSize, ds.RasterYSize


def block_world_grid(gt, xoff, yoff, xsize, ysize):
    cols, rows = np.meshgrid(
        np.arange(xoff, xoff + xsize),
        np.arange(yoff, yoff + ysize)
    )

    x = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    y = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]

    return x, y


def save_npy(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--dsm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--epsg", required=True)

    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--min-valid-pixels", type=int, default=500)

    args = parser.parse_args()

    project_dir = Path(args.project)
    image_dir = project_dir / "images"
    dense_dir = Path(args.dense)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    cameras_txt = dense_dir / "sparse" / "cameras.txt"
    images_txt = dense_dir / "sparse" / "images.txt"

    if not cameras_txt.exists():
        raise FileNotFoundError(f"Missing camera parameter file: {cameras_txt}")

    if not images_txt.exists():
        raise FileNotFoundError(f"Missing camera pose file: {images_txt}")

    cameras = read_cameras_txt(cameras_txt)
    images = read_images_txt(images_txt)

    dsm_ds, dsm_band, gt, nodata, dsm_width, dsm_height = get_dsm_info(args.dsm)

    print("DSM size:", dsm_width, "x", dsm_height)
    print("DSM geotransform:", gt)
    print("Number of cameras:", len(cameras))
    print("Number of images:", len(images))

    master_index = []

    for image_name, image_rec in images.items():

        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"Skipping missing source image: {image_path}")
            continue

        camera_rec = cameras[image_rec["camera_id"]]

        print(f"Processing image projection for: {image_name}")
        print(f"Camera model: {camera_rec['model']}")

        image_tile_count = 0

        for yoff in range(0, dsm_height, args.block_size):
            ysize = min(args.block_size, dsm_height - yoff)

            for xoff in range(0, dsm_width, args.block_size):
                xsize = min(args.block_size, dsm_width - xoff)

                z_block = dsm_band.ReadAsArray(xoff, yoff, xsize, ysize)

                if z_block is None:
                    continue

                z_block = z_block.astype(np.float64)
                valid_z = z_block != nodata

                if not np.any(valid_z):
                    continue

                world_x, world_y = block_world_grid(
                    gt,
                    xoff,
                    yoff,
                    xsize,
                    ysize
                )

                points = np.column_stack([
                    world_x.reshape(-1),
                    world_y.reshape(-1),
                    z_block.reshape(-1)
                ])

                valid_flat = valid_z.reshape(-1)
                points_valid = points[valid_flat]

                if points_valid.shape[0] == 0:
                    continue

                u, v, inside = project_points_to_image(
                    points_valid,
                    image_rec,
                    camera_rec
                )

                if np.count_nonzero(inside) < args.min_valid_pixels:
                    continue

                xmap_flat = np.full((ysize * xsize,), -1, dtype=np.float32)
                ymap_flat = np.full((ysize * xsize,), -1, dtype=np.float32)

                valid_indices = np.where(valid_flat)[0]
                selected_indices = valid_indices[inside]

                xmap_flat[selected_indices] = u[inside]
                ymap_flat[selected_indices] = v[inside]

                xmap = xmap_flat.reshape(ysize, xsize)
                ymap = ymap_flat.reshape(ysize, xsize)

                base = Path(image_name).stem
                tile_id = f"{base}_x{xoff}_y{yoff}_w{xsize}_h{ysize}"

                tile_dir = output_dir / base
                xmap_path = tile_dir / f"{tile_id}_xmap.npy"
                ymap_path = tile_dir / f"{tile_id}_ymap.npy"
                json_path = tile_dir / f"{tile_id}.json"

                save_npy(xmap_path, xmap)
                save_npy(ymap_path, ymap)

                ulx = gt[0] + xoff * gt[1]
                uly = gt[3] + yoff * gt[5]
                lrx = gt[0] + (xoff + xsize) * gt[1]
                lry = gt[3] + (yoff + ysize) * gt[5]

                meta = {
                    "image_path": str(image_path),
                    "camera_model": camera_rec["model"],
                    "camera_params": camera_rec["params"],
                    "camera_id": image_rec["camera_id"],
                    "xmap": str(xmap_path),
                    "ymap": str(ymap_path),
                    "width": int(xsize),
                    "height": int(ysize),
                    "dsm_xoff": int(xoff),
                    "dsm_yoff": int(yoff),
                    "ulx": float(ulx),
                    "uly": float(uly),
                    "lrx": float(lrx),
                    "lry": float(lry),
                    "epsg": args.epsg
                }

                with open(json_path, "w") as f:
                    json.dump(meta, f, indent=2)

                master_index.append(str(json_path))
                image_tile_count += 1

        print(f"Generated {image_tile_count} remap tiles for {image_name}")

    index_path = output_dir / "remap_index.json"

    with open(index_path, "w") as f:
        json.dump(master_index, f, indent=2)

    print("Finished generating remap tiles.")
    print("Index file:", index_path)


if __name__ == "__main__":
    main()
