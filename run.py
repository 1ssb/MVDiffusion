import torch
import argparse
import yaml
from src.lightning_pano_gen import PanoGenerator
import numpy as np
import cv2
import os
from PIL import Image
import generate_video_tool.lib.multi_Perspec2Equirec as m_P2E
import time
from datetime import datetime

torch.manual_seed(0)


def get_K_R(FOV, THETA, PHI, height, width):
    """
    Build camera intrinsics K and rotation R for a single view.
    """
    f = 0.5 * width / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float32)

    # First rotate around Y by THETA, then rotate that result around X by PHI.
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues((R1 @ x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


def resize_and_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize the input image so that its shorter side = size, then center‐crop to (size × size).
    """
    H, W, _ = img.shape
    if H == W:
        return cv2.resize(img, (size, size))
    if H > W:
        new_h = int(size * H / W)
        resized = cv2.resize(img, (size, new_h))
        top = (new_h - size) // 2
        return resized[top : top + size, :]
    else:
        new_w = int(size * W / H)
        resized = cv2.resize(img, (new_w, size))
        left = (new_w - size) // 2
        return resized[:, left : left + size]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_image",
        type=str,
        required=True,
        help="Path to the input image (will be resized and center-cropped)."
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the scene."
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate in one batch."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.num_samples

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Load config + instantiate model
    config_file = "configs/pano_generation.yaml"
    config = yaml.load(open(config_file, "rb"), Loader=yaml.SafeLoader)
    model = PanoGenerator(config).cuda().eval()

    # Load checkpoint, filtering keys to match PanoGenerator exactly
    ckpt = torch.load("weights/pano.ckpt", map_location="cpu")
    ckpt_state = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        name = k.replace("model.", "") if k.startswith("model.") else k
        if name in model_state and v.shape == model_state[name].shape:
            filtered[name] = v
    model.load_state_dict(filtered, strict=False)

    resolution = config["dataset"]["resolution"]

    # Precompute a single set of Ks and Rs for the 8 fixed views,
    # then broadcast to batch size n
    Ks_single = []
    Rs_single = []
    for i in range(8):
        angle = (45 * i) % 360
        K, R = get_K_R(
            FOV=90,
            THETA=angle,
            PHI=0,
            height=resolution,
            width=resolution,
        )
        Ks_single.append(K)
        Rs_single.append(R)

    # Stack into numpy arrays (8, 3, 3), then tile to (n, 8, 3, 3)
    Ks_np = np.stack(Ks_single, axis=0)               # shape: (8, 3, 3)
    Rs_np = np.stack(Rs_single, axis=0)               # shape: (8, 3, 3)
    Ks = torch.from_numpy(Ks_np).unsqueeze(0)          # shape: (1, 8, 3, 3)
    Rs = torch.from_numpy(Rs_np).unsqueeze(0)          # shape: (1, 8, 3, 3)
    Ks = Ks.repeat(n, 1, 1, 1).float().cuda()          # shape: (n, 8, 3, 3)
    Rs = Rs.repeat(n, 1, 1, 1).float().cuda()          # shape: (n, 8, 3, 3)

    # Read + preprocess the input image once
    raw = cv2.imread(args.input_image)
    if raw is None:
        raise FileNotFoundError(f"Cannot load image at {args.input_image}")
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    cropped = resize_and_center_crop(rgb, resolution)

    # Build a single tensor of shape (n, 8, H, W, 3), dtype=uint8, CPU→GPU
    images = torch.zeros((n, 8, resolution, resolution, 3), dtype=torch.uint8).cuda()
    for batch_i in range(n):
        images[batch_i, 0] = torch.from_numpy(cropped)  # place same image into view‐0 of each sample

    # Build `prompt` as a list of length 8, where each element is itself a list of n strings
    # i.e. prompt[view_i] = [prompt for sample0, prompt for sample1, ..., prompt for sample_{n-1}]
    prompt_batched = [[args.prompt] * n for _ in range(8)]

    # Save the prompt.txt once
    with open(os.path.join(run_dir, "prompt.txt"), "w") as f:
        f.write(args.prompt)

    # Build the batch‐dict and call inference in one shot
    batch = {
        "images": images,                # shape: (n, 8, H, W, 3)
        "prompt": prompt_batched,        # length 8, each is length-n list
        "R": Rs,                         # shape: (n, 8, 3, 3)
        "K": Ks,                         # shape: (n, 8, 3, 3)
    }

    # Run inference once for all n samples
    t0 = time.time()
    with torch.no_grad():
        preds = model.inference(batch)   # preds shape: (n, 8, H, W, 3)
    total_time = time.time() - t0
    print(f"Ran inference on {n} samples in {total_time:.2f}s")

    # Loop over each of the n outputs
    for sample_i in range(n):
        sample_dir = os.path.join(run_dir, f"sample_{sample_i}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save each of the 8 view images
        view_paths = []
        for view_idx in range(8):
            arr = preds[sample_i, view_idx]             # (H, W, 3) numpy‐style array
            img_pil = Image.fromarray(arr)
            path = os.path.join(sample_dir, f"view_{view_idx}.png")
            img_pil.save(path)
            view_paths.append(path)

        # Build equirect panorama out of the 8 saved views
        perspectives = [cv2.imread(p) for p in view_paths]
        angles = [[90, 45 * i, 0] for i in range(8)]
        ee = m_P2E.Perspective(perspectives, angles)

        # Save a 512×1024 pano
        pano_small = ee.GetEquirec(512, 1024)
        cv2.imwrite(os.path.join(sample_dir, "pano_512x1024.png"), pano_small)

        # Save a 1024×2048 pano
        pano_large = ee.GetEquirec(1024, 2048)
        cv2.imwrite(os.path.join(sample_dir, "pano_1024x2048.png"), pano_large)

    print(f"\nAll outputs saved under: {run_dir}")
