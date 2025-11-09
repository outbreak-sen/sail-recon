import argparse
import os
import torch
from tqdm import tqdm

from eval.utils.device import to_cpu
from eval.utils.eval_utils import uniform_sample
from sailrecon.models.sail_recon import SailRecon
from sailrecon.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def demo(args):
    # ----------------- Load model -----------------
    _URL = "https://huggingface.co/HKUST-SAIL/SAIL-Recon/resolve/main/sailrecon.pt"
    model = SailRecon(kv_cache=True)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(_URL, model_dir=args.ckpt)
        )
    model = model.to(device=device)
    model.eval()

    # ----------------- Load images or video -----------------
    scene_name = "1"
    if args.vid_dir is not None:
        import cv2
        video_path = args.vid_dir
        vs = cv2.VideoCapture(video_path)
        tmp_dir = os.path.join("tmp_video", os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(tmp_dir, exist_ok=True)
        image_names = []
        count = 0
        frame_id = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            # 抽帧：每隔 frame_interval 帧取一帧
            if count % args.frame_interval == 0:
                image_path = os.path.join(tmp_dir, f"{frame_id:06}.png")
                cv2.imwrite(image_path, frame)
                image_names.append(image_path)
                frame_id += 1
            count += 1
        vs.release()

        # 限制最大帧数
        if args.max_frames > 0 and len(image_names) > args.max_frames:
            image_names = image_names[: args.max_frames]

        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.splitext(os.path.basename(video_path))[0]

    else:
        image_names = sorted(
            [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir) if f.lower().endswith((".jpg", ".png"))]
        )

        # 抽帧与帧数限制
        image_names = image_names[:: args.frame_interval]
        if args.max_frames > 0 and len(image_names) > args.max_frames:
            image_names = image_names[: args.max_frames]

        images = load_and_preprocess_images(image_names).to(device)
        scene_name = os.path.basename(args.img_dir)

    if len(image_names) == 0:
        raise ValueError("No images found after sampling. Check --img_dir/--vid_dir and --frame_interval / --max_frames.")

    # ----------------- Anchor Image Selection -----------------
    select_indices = uniform_sample(len(image_names), min(100, len(image_names)))
    anchor_images = images[select_indices]
    os.makedirs(os.path.join(args.out_dir, scene_name), exist_ok=True)

    # ----------------- Reconstruction -----------------
    with torch.no_grad():
        # 新的 autocast 用法，避免 future warning
        device_type = "cuda" if device == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            print(f"Processing {len(anchor_images)} anchor images ...")
            model.tmp_forward(anchor_images)
            # remove global blocks to save memory
            if hasattr(model, "aggregator") and hasattr(model.aggregator, "global_blocks"):
                try:
                    del model.aggregator.global_blocks
                except Exception:
                    pass

            predictions = []
            batch_size = 20  # 每批推理，防止显存爆炸

            with tqdm(total=len(image_names), desc="Relocalizing") as pbar:
                # iterate by splitting images into batches (works even if last batch smaller)
                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    # 使用 memory_save 模式以降低显存（你可以根据需要切换 True/False）
                    preds = model.reloc(batch, memory_save=True)
                    # preds 可能是 list of dicts
                    if isinstance(preds, list):
                        predictions += to_cpu(preds)
                    else:
                        # 保险处理（如果 model.reloc 返回 tensor 等）
                        try:
                            predictions += to_cpu(list(preds))
                        except Exception:
                            # 兜底：把 preds 当成单元素
                            predictions.append(to_cpu(preds))
                    pbar.update(batch.size(0))

            # ----------------- Save Results（更鲁棒） -----------------
            from eval.utils.geometry import save_pointcloud_with_plyfile
            from eval.utils.eval_utils import save_kitti_poses
            import numpy as np

            # 过滤出包含 point_map_by_unprojection 的预测（有些帧可能缺失）
            preds_with_points = []
            missing_point_idxs = []
            for idx, one in enumerate(predictions):
                if isinstance(one, dict) and "point_map_by_unprojection" in one and one["point_map_by_unprojection"] is not None:
                    preds_with_points.append(one)
                else:
                    missing_point_idxs.append(idx)

            if len(preds_with_points) == 0:
                print("Warning: no predictions contain 'point_map_by_unprojection'. Skipping point cloud save.")
            else:
                if len(missing_point_idxs) > 0:
                    print(f"Warning: {len(missing_point_idxs)} frames skipped for point cloud because 'point_map_by_unprojection' missing. Indices sample: {missing_point_idxs[:10]}")
                try:
                    save_pointcloud_with_plyfile(preds_with_points, os.path.join(args.out_dir, scene_name, "pred.ply"))
                except Exception as e:
                    print(f"Error when saving point cloud: {e}")

            # 保存 poses：只使用包含 extrinsic 的预测
            poses_w2c_estimated = []
            for one in predictions:
                if isinstance(one, dict) and "extrinsic" in one and one["extrinsic"] is not None:
                    try:
                        # extrinsic might have shape like (1,4,4) or similar
                        mat = one["extrinsic"][0].cpu().numpy()
                        poses_w2c_estimated.append(mat)
                    except Exception:
                        continue

            if len(poses_w2c_estimated) == 0:
                print("Warning: no valid extrinsic poses found; skipping pose file save.")
            else:
                poses_c2w_estimated = [np.linalg.inv(np.vstack([pose, np.array([0, 0, 0, 1])])) for pose in poses_w2c_estimated]
                save_kitti_poses(poses_c2w_estimated, os.path.join(args.out_dir, scene_name, "pred.txt"))

    print(f"\n✅ Done. Results saved to {os.path.join(args.out_dir, scene_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="samples/kitchen", help="input image folder")
    parser.add_argument("--vid_dir", type=str, default=None, help="input video path")
    parser.add_argument("--out_dir", type=str, default="outputs", help="output folder")
    parser.add_argument("--ckpt", type=str, default=None, help="pretrained model checkpoint")

    # 抽帧参数
    parser.add_argument("--frame_interval", type=int, default=1, help="frame sampling interval (e.g. 5 = every 5th frame)")
    parser.add_argument("--max_frames", type=int, default=-1, help="maximum number of frames to process (-1 = no limit)")

    args = parser.parse_args()
    demo(args)
