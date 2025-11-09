import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from eval.utils.geometry import save_pointcloud_with_plyfile
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images
from scipy.spatial.transform import Rotation as R
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_images_from_dir(img_dir, frame_interval=1, max_frames=None):
    image_names = sorted(os.listdir(img_dir))
    image_names = [f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_names = image_names[::frame_interval]
    if max_frames is not None:
        image_names = image_names[:max_frames]
    full_paths = [os.path.join(img_dir, f) for f in image_names]
    return full_paths


def demo(args):
    # Initialize model
    model = SailRecon(kv_cache=True)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt",
                model_dir=args.ckpt
            )
        )
    model = model.to(device=device)
    model.eval()

    # Load input images
    image_names = load_images_from_dir(args.img_dir, args.frame_interval, args.max_frames)
    images = load_and_preprocess_images(image_names).to(device)
    scene_name = os.path.basename(args.img_dir.rstrip("/"))

    # anchor images selection
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]

    os.makedirs(os.path.join(args.out_dir, scene_name), exist_ok=True)

    # model inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            print(f"Processing {len(anchor_images)} anchor images ...")
            model.tmp_forward(anchor_images)
            del model.aggregator.global_blocks

            predictions = []
            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                for img_split in images.split(args.batch_size, dim=0):
                    preds = model.reloc(img_split, memory_save=False)
                    predictions += to_cpu(preds)
                    pbar.update(img_split.shape[0])

    # ---------- Save results ----------
    output_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 输出文件命名 ----------
    if args.tum:
        # 从路径中提取倒数第二层目录名
        dataset_name = Path(args.img_dir).resolve().parent.name
        traj_file = os.path.join(output_dir, f"traj_{dataset_name}.txt")
        ply_file = os.path.join(output_dir, f"{dataset_name}.ply")
    else:
        dataset_name = Path(args.img_dir).resolve().name
        traj_file = os.path.join(output_dir, f"traj_{dataset_name}.txt")
        ply_file = os.path.join(output_dir, f"{dataset_name}.ply")

    print(f"Saving trajectory: {traj_file}")
    print(f"Saving point cloud: {ply_file}")

    # ---------- Pose extraction ----------
    poses_w2c_estimated = [r["extrinsic"][0].cpu().numpy() for r in predictions]
    poses_c2w = [np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])])) for pose in poses_w2c_estimated]

    # ---------- Save trajectory (TUM format always) ----------
    with open(traj_file, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for img_name, pose in zip(image_names, poses_c2w):
            t = pose[:3, 3]
            R_mat = pose[:3, :3]
            q = R.from_matrix(R_mat).as_quat()  # (x,y,z,w)
            timestamp = os.path.splitext(os.path.basename(img_name))[0]
            f.write(f"{timestamp} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"[✔] Trajectory saved: {traj_file}")

    # ---------- Save point cloud ----------
    try:
        save_pointcloud_with_plyfile(predictions, ply_file)
        print(f"[✔] Point cloud saved: {ply_file}")
    except Exception as e:
        print(f"[⚠] Failed to save point cloud: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="input image folder")
    parser.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model path")
    parser.add_argument("--frame_interval", type=int, default=1, help="frame sampling interval")
    parser.add_argument("--max_frames", type=int, default=None, help="maximum frames to process")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--tum", action="store_true", help="enable TUM naming convention")
    parser.add_argument("--debug_dump", action="store_true", help="enable debug output")
    args = parser.parse_args()
    demo(args)
