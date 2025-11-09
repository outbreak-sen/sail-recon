#!/usr/bin/env python3
"""
demo_traj_and_ply.py

增强版 demo：
- 支持 --frame_interval / --max_frames 控制（从你的请求）
- 保存 KITTI 风格 pred.txt（每行 12 个数字）并保存 pred_index_to_image.txt (index->image path)
- 尝试从 predictions 中读取 point_map_by_unprojection；若缺失，尝试使用 model 的 camera/depth head 生成点云并合并保存为 PLY
- 更鲁棒的 batch 处理与日志
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def unproject_depth_to_points(depth_map, intrinsic, extrinsic_w2c):
    """
    depth_map: H x W tensor (camera depth along z in camera frame)
    intrinsic: 3x3 numpy or tensor matrix (fx 0 cx; 0 fy cy; 0 0 1)
    extrinsic_w2c: 3x4 numpy matrix, world -> camera (W2C). We'll invert to get C2W.
    Returns: Nx3 numpy array of points in world coordinates
    """
    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.cpu().numpy()
    else:
        depth = depth_map
    if isinstance(intrinsic, torch.Tensor):
        K = intrinsic.cpu().numpy()
    else:
        K = intrinsic
    if isinstance(extrinsic_w2c, torch.Tensor):
        ext = extrinsic_w2c.cpu().numpy()
    else:
        ext = extrinsic_w2c

    # invert extrinsic to get camera -> world (C2W)
    # ext is 3x4 (R | t) mapping world -> camera: x_c = R * x_w + t
    # build 4x4:
    ext4 = np.vstack([ext, np.array([0.0, 0.0, 0.0, 1.0])])
    c2w = np.linalg.inv(ext4)  # 4x4

    H, W = depth.shape[-2], depth.shape[-1] if depth.ndim == 2 else depth.shape
    # create pixel grid: u in [0,W), v in [0,H)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')  # ys: HxW, xs: HxW

    xs = xs.reshape(-1).astype(np.float32)
    ys = ys.reshape(-1).astype(np.float32)
    zs = depth.reshape(-1).astype(np.float32)

    # mask invalid depths (<=0)
    keep = zs > 0
    if keep.sum() == 0:
        return np.zeros((0,3), dtype=np.float32)

    xs = xs[keep]
    ys = ys[keep]
    zs = zs[keep]

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    x_cam = (xs - cx) / fx * zs
    y_cam = (ys - cy) / fy * zs
    z_cam = zs

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)  # N x 4
    # transform to world
    pts_world = (c2w @ pts_cam.T).T[:, :3]  # N x 3
    return pts_world

def save_kitti_poses_from_list(poses_c2w_list, out_path):
    """
    poses_c2w_list: list of 3x4 numpy arrays (camera->world)
    Saves KITTI-style pose text file: each line 12 numbers (row-major of 3x4)
    """
    with open(out_path, "w") as f:
        for p in poses_c2w_list:
            arr = np.asarray(p).reshape(3,4)
            flat = arr.reshape(-1)
            line = " ".join([f"{x:.6f}" for x in flat])
            f.write(line + "\n")

def save_index_map(image_paths, out_path):
    with open(out_path, "w") as f:
        for i,p in enumerate(image_paths):
            f.write(f"{i} {p}\n")

def save_ply(points, out_path, max_points=500000):
    """
    points: Nx3 numpy array
    Writes a simple ascii PLY
    """
    if points.shape[0] == 0:
        print("No points to save to PLY.")
        return
    pts = points
    # subsample if too many
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    n = pts.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}\n")
    print(f"Saved PLY with {n} points to {out_path}")

def demo(args):
    # load model
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model = SailRecon(kv_cache=True)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    else:
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, model_dir=args.ckpt))
    model = model.to(device=device)
    model.eval()

    # prepare images list (with frame interval & max_frames)
    scene_name = "scene"
    if args.vid_dir is not None:
        import cv2
        vs = cv2.VideoCapture(args.vid_dir)
        tmp_dir = os.path.join("tmp_video", os.path.splitext(os.path.basename(args.vid_dir))[0])
        os.makedirs(tmp_dir, exist_ok=True)
        image_names = []
        cnt = 0
        saved_idx = 0
        while True:
            ok, frame = vs.read()
            if not ok:
                break
            if cnt % args.frame_interval == 0:
                img_path = os.path.join(tmp_dir, f"{saved_idx:06d}.png")
                cv2.imwrite(img_path, frame)
                image_names.append(img_path)
                saved_idx += 1
            cnt += 1
        vs.release()
        if args.max_frames > 0 and len(image_names) > args.max_frames:
            image_names = image_names[: args.max_frames]
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.splitext(os.path.basename(args.vid_dir))[0]
    else:
        imgs = sorted([os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        imgs = imgs[:: args.frame_interval]
        if args.max_frames > 0 and len(imgs) > args.max_frames:
            imgs = imgs[: args.max_frames]
        image_names = imgs
        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(args.img_dir)

    assert len(image_names) > 0, "No images found after sampling. Check inputs."

    out_scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)

    # anchor selection (same strategy as original)
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]

    # run anchor processing and relocalization to get predictions as before
    predictions = []
    device_type = "cuda" if device == "cuda" else "cpu"
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            print("Processing anchor images ...")
            # try to use tmp_forward if available
            try:
                model.tmp_forward(anchor_images)
            except Exception as e:
                print("Warning: model.tmp_forward failed or unavailable:", e)
                # Some model versions may use aggregator directly
                try:
                    _ = model.aggregator(anchor_images)
                except Exception:
                    pass

            # remove global blocks if present to save memory
            try:
                if hasattr(model, "aggregator") and hasattr(model.aggregator, "global_blocks"):
                    del model.aggregator.global_blocks
            except Exception:
                pass

            # relocalize (batch-wise)
            batch_size = args.batch_size
            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    # call reloc - in many implementations this returns a list of dicts
                    try:
                        preds = model.reloc(batch, memory_save=True)
                    except Exception as e:
                        print("model.reloc failed on batch; error:", e)
                        # try model.reloc(batch, memory_save=False)
                        try:
                            preds = model.reloc(batch, memory_save=False)
                        except Exception as e2:
                            print("model.reloc retry also failed:", e2)
                            preds = []
                    # accumulate
                    if isinstance(preds, list):
                        predictions += to_cpu(preds)
                    else:
                        # attempt to coerce
                        try:
                            predictions += to_cpu(list(preds))
                        except Exception:
                            predictions.append(to_cpu(preds))
                    pbar.update(batch.size(0))

    # predictions list should correspond to image_names order (or close). We'll try to align by index.
    # Now: build poses list (C2W) for all frames we have extrinsic for
    poses_c2w = []
    missing_pose_indices = []
    for idx, one in enumerate(predictions):
        try:
            if isinstance(one, dict) and "extrinsic" in one and one["extrinsic"] is not None:
                pose_w2c = one["extrinsic"][0].cpu().numpy()  # 3x4 expected (W2C)
                # invert to C2W
                ext4 = np.vstack([pose_w2c, np.array([0.0,0.0,0.0,1.0])])
                c2w = np.linalg.inv(ext4)[:3,:]  # 3x4
                poses_c2w.append(c2w)
            else:
                missing_pose_indices.append(idx)
        except Exception as e:
            missing_pose_indices.append(idx)

    # Save index->image mapping
    save_index_map(image_names, os.path.join(out_scene_dir, "pred_index_to_image.txt"))

    # Save KITTI poses (only those frames we have). NOTE: order is as poses_c2w list - which matches frames for which predictions had extrinsic.
    if len(poses_c2w) > 0:
        save_kitti_poses_from_list(poses_c2w, os.path.join(out_scene_dir, "pred.txt"))
        print(f"Saved {len(poses_c2w)} poses to pred.txt (skipped {len(missing_pose_indices)} frames without extrinsic).")
    else:
        print("No valid extrinsic poses found in predictions; pred.txt not saved.")

    # Build / collect point clouds:
    all_points = []

    # 1) First try: use predictions' point_map_by_unprojection if present
    for idx, one in enumerate(predictions):
        try:
            if isinstance(one, dict) and "point_map_by_unprojection" in one and one["point_map_by_unprojection"] is not None:
                pm = one["point_map_by_unprojection"]  # expect shape (1,H,W,3) or (H,W,3)
                # normalize to numpy Nx3
                if isinstance(pm, torch.Tensor):
                    pm_np = pm.cpu().numpy()
                else:
                    pm_np = np.array(pm)
                # handle batch dim
                if pm_np.ndim == 4 and pm_np.shape[0] == 1:
                    pm_np = pm_np[0]
                h,w, c = pm_np.shape
                pts = pm_np.reshape(-1,3)
                # optionally filter invalid points (e.g., zeros)
                mask = ~np.isnan(pts).any(axis=1)
                pts = pts[mask]
                if pts.shape[0] > 0:
                    all_points.append(pts)
        except Exception as e:
            continue

    # 2) Second try: use depth + intrinsics + extrinsic (compute unprojection)
    # For frames that didn't have point_map_by_unprojection, try to compute using depth & extrinsic/intrinsic:
    for idx, one in enumerate(predictions):
        # skip if we already used point_map_by_unprojection (we can detect by checking presence)
        used = False
        if isinstance(one, dict) and "point_map_by_unprojection" in one and one["point_map_by_unprojection"] is not None:
            used = True
        if used:
            continue
        # need depth and intrinsic and extrinsic
        try:
            has_depth = isinstance(one, dict) and ("depth_map" in one or "depth" in one)
            has_extrinsic = isinstance(one, dict) and ("extrinsic" in one and one["extrinsic"] is not None)
            # Try common keys
            depth_tensor = None
            if isinstance(one, dict):
                if "depth_map" in one and one["depth_map"] is not None:
                    depth_tensor = one["depth_map"]
                elif "depth" in one and one["depth"] is not None:
                    depth_tensor = one["depth"]
            if depth_tensor is None or not has_extrinsic:
                # can't build from this prediction; skip
                continue
            # get extrinsic W2C
            pose_w2c = one["extrinsic"][0].cpu().numpy()
            # try to get intrinsic from prediction if present
            intrinsic = None
            if "intrinsic" in one and one["intrinsic"] is not None:
                intrinsic = one["intrinsic"][0].cpu().numpy()
            # If intrinsic missing, try to extract from model by running aggregator + camera_head on that image
            if intrinsic is None:
                # build single-image tensor and call model heads
                try:
                    img_tensor = load_and_preprocess_images([image_names[idx]]).to(device)
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device_type, dtype=dtype):
                            # aggregated tokens
                            try:
                                aggregated_tokens_list, ps_idx = model.aggregator(img_tensor)
                            except Exception:
                                # some versions return only one output
                                aggregated_tokens_list = model.aggregator(img_tensor)
                                ps_idx = None
                            # camera head
                            try:
                                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                                # try to call pose encoding util if available
                                from sailrecon.utils.pose_enc import pose_encoding_to_extri_intri
                                extrinsic_tmp, intrinsic_tmp = pose_encoding_to_extri_intri(pose_enc, img_tensor.shape[-2:])
                                intrinsic = intrinsic_tmp[0].cpu().numpy()
                                # overwrite pose_w2c if not present
                                if pose_w2c is None:
                                    pose_w2c = extrinsic_tmp[0].cpu().numpy()
                            except Exception as e:
                                # camera head failed
                                intrinsic = None
                except Exception:
                    intrinsic = None

            if intrinsic is None:
                # cannot compute without intrinsics; skip
                continue

            # depth tensor may have batch dim etc; get HxW numpy
            if isinstance(depth_tensor, torch.Tensor):
                dt = depth_tensor.cpu().numpy()
            else:
                dt = np.array(depth_tensor)
            # handle shapes: if shape (1,1,H,W) or (1,H,W) or (H,W)
            if dt.ndim == 4:
                # assume (1,1,H,W)
                dt = dt.squeeze()
            if dt.ndim == 3 and dt.shape[0] == 1:
                dt = dt.squeeze(0)
            if dt.ndim == 3 and dt.shape[-1] == 3:
                # unlikely: it's a point_map; skip here
                continue
            # now dt should be H x W
            pts_world = unproject_depth_to_points(dt, intrinsic, pose_w2c)
            if pts_world.shape[0] > 0:
                all_points.append(pts_world)
        except Exception as e:
            # ignore per-frame issues
            continue

    # 3) Third fallback: if above fails, try to compute depth/intrinsic for whole image set using aggregator/depth_head
    if len(all_points) == 0:
        print("No points extracted from predictions directly. Attempting a full pass: calling model.aggregator + camera_head + depth_head over all images (batched).")
        batch_size = args.batch_size
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                for i in range(0, len(images), batch_size):
                    batch = images[i:i+batch_size]
                    try:
                        # aggregated tokens
                        aggregated_tokens_list, ps_idx = model.aggregator(batch)
                    except Exception:
                        try:
                            aggregated_tokens_list = model.aggregator(batch)
                            ps_idx = None
                        except Exception as e:
                            print("Aggregator failed on batch:", e)
                            continue
                    # camera encoding
                    try:
                        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                        # convert to extrinsic, intrinsic if util exists
                        try:
                            from sailrecon.utils.pose_enc import pose_encoding_to_extri_intri
                            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
                        except Exception:
                            extrinsic = None
                            intrinsic = None
                        # depth
                        try:
                            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batch, ps_idx)
                        except Exception:
                            depth_map = None
                        # if we have depth and extrinsic/intrinsic, unproject per frame
                        if depth_map is not None and extrinsic is not None and intrinsic is not None:
                            # depth_map shape likely (B,1,H,W) or (B,H,W)
                            for bi in range(depth_map.shape[0]):
                                d = depth_map[bi]
                                ext = extrinsic[bi].cpu().numpy()
                                K = intrinsic[bi].cpu().numpy()
                                # normalize d to numpy HxW
                                dn = d.squeeze().cpu().numpy() if isinstance(d, torch.Tensor) else np.array(d).squeeze()
                                pts = unproject_depth_to_points(dn, K, ext)
                                if pts.shape[0] > 0:
                                    all_points.append(pts)
                    except Exception as e:
                        print("Batch processing failed for point cloud:", e)
                        continue

    # merge and save point cloud
    if len(all_points) > 0:
        pts_all = np.concatenate(all_points, axis=0)
        print(f"Total extracted points before downsample: {pts_all.shape[0]}")
        save_ply(pts_all, os.path.join(out_scene_dir, "pred.ply"), max_points=args.max_ply_points)
    else:
        print("No points collected for PLY (no point_map_by_unprojection or depth/intrinsic available).")

    # optional debug dump
    if args.debug_dump:
        try:
            simple_preds = []
            for p in predictions:
                # attempt to shrink large tensors (just record keys and shapes)
                if isinstance(p, dict):
                    rec = {}
                    for k,v in p.items():
                        try:
                            if torch.is_tensor(v):
                                rec[k] = f"tensor{list(v.shape)}"
                            else:
                                rec[k] = str(type(v))
                        except Exception:
                            rec[k] = str(type(v))
                    simple_preds.append(rec)
                else:
                    simple_preds.append(str(type(p)))
            with open(os.path.join(out_scene_dir, "pred_debug_predictions.json"), "w") as f:
                json.dump(simple_preds, f, indent=2)
            print("Saved pred_debug_predictions.json for inspection.")
        except Exception as e:
            print("Failed to dump debug predictions:", e)

    print("Done. Outputs in:", out_scene_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=None, help="input image folder (mutually exclusive with --vid_dir)")
    parser.add_argument("--vid_dir", type=str, default=None, help="input video path")
    parser.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model checkpoint")
    parser.add_argument("--frame_interval", type=int, default=1, help="frame sampling interval")
    parser.add_argument("--max_frames", type=int, default=-1, help="maximum frames to process (-1 = no limit)")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for relocalization / depth inference")
    parser.add_argument("--max_ply_points", type=int, default=500000, help="max points to write in PLY (subsample if larger)")
    parser.add_argument("--debug_dump", action="store_true", help="dump simplified predictions to json for debug")
    args = parser.parse_args()
    demo(args)
