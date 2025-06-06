#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import argparse
import yaml
import numpy as np
import cv2
import os
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.dataset import MP3Ddataset, Scannetdataset
from src.lightning_pano_gen import PanoGenerator
from src.lightning_pano_outpaint import PanoOutpaintGenerator
from src.lightning_depth import DepthGenerator
import generate_video_tool.lib.multi_Perspec2Equirec as m_P2E
import time

torch.manual_seed(0)


def get_K_R(FOV, THETA, PHI, H, W):
    """
    Build intrinsics K and rotation R for a camera with given FOV and spherical angles.
    Returns:
        K: float32 numpy array of shape (3,3)
        R: float32 numpy array of shape (3,3)
    """
    f = 0.5 * W / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues((R1 @ x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


def resize_and_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize the input image so that the shorter side = size, then center-crop to (size × size).
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


def filter_state_dict(checkpoint_state: dict, model_state: dict) -> dict:
    """
    Keep only those keys in checkpoint_state whose names (after stripping "model.")
    match exactly a key in model_state, and whose tensor shapes agree.
    """
    filtered = {}
    for k, v in checkpoint_state.items():
        name = k.replace("model.", "") if k.startswith("model.") else k
        if name in model_state and v.shape == model_state[name].shape:
            filtered[name] = v
    return filtered


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "main_cfg_path",
        type=str,
        help="Path to main config file (e.g. pano_generation_outpaint.yaml)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default_exp",
        help="Experiment name (also used as output subfolder).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of single-image inferences to run (when using --input_image).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (when using dataset).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (e.g. weights/pano_outpaint.ckpt).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Whether to run validation or test on the dataset.",
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="If set, run evaluation on the training split instead of validation.",
    )

    # If provided, skip dataset loading and run on single-image batch inference
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to a single skybox image. If given, will bypass dataset and run inference multiple times.",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    # Load main YAML config
    config = yaml.load(open(args.main_cfg_path, "rb"), Loader=yaml.SafeLoader)
    if hasattr(args, "max_epochs"):
        config["train"]["max_epochs"] = args.max_epochs
    elif "max_epochs" not in config["train"]:
        config["train"]["max_epochs"] = 1

    # Instantiate the appropriate Lightning model
    model_type = config["model"]["model_type"]
    if model_type == "pano_generation":
        model = PanoGenerator(config)
    elif model_type == "pano_generation_outpaint":
        model = PanoOutpaintGenerator(config)
    elif model_type == "depth":
        model = DepthGenerator(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load checkpoint if provided, applying filtering to avoid missing/unexpected key errors
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        ckpt_state = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = filter_state_dict(ckpt_state, model_state)
        model.load_state_dict(filtered, strict=False)

    # If input_image is provided, run multiple single-image inferences
    if args.input_image is not None:
        resolution = config["dataset"]["resolution"]
        out_base = os.path.join("outputs", args.exp_name)
        os.makedirs(out_base, exist_ok=True)

        raw = cv2.imread(args.input_image)
        if raw is None:
            raise FileNotFoundError(f"Cannot load image at {args.input_image}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        cropped = resize_and_center_crop(rgb, resolution)
        cropped_float = cropped.astype(np.float32) / 255.0

        # Precompute K and R once
        Ks = []
        Rs = []
        for i in range(8):
            yaw = (45 * i) % 360
            Kmat, Rmat = get_K_R(
                FOV=90,
                THETA=yaw,
                PHI=0,
                H=resolution,
                W=resolution,
            )
            Ks.append(Kmat)
            Rs.append(Rmat)
        K_tensor = torch.from_numpy(np.stack(Ks, axis=0)).unsqueeze(0).float().cuda()  # (1,8,3,3)
        R_tensor = torch.from_numpy(np.stack(Rs, axis=0)).unsqueeze(0).float().cuda()  # (1,8,3,3)

        prompt_list = [""] * 8  # replace with real text if needed

        model = model.cuda().eval()
        for si in range(args.batch_size):
            sample_dir = os.path.join(out_base, f"sample_{si}")
            os.makedirs(sample_dir, exist_ok=True)

            # Build batch of shape (1,8,res,res,3)
            images_tensor = torch.zeros((1, 8, resolution, resolution, 3), dtype=torch.float32).cuda()
            images_tensor[0, 0] = torch.from_numpy(cropped_float)

            batch = {
                "images": images_tensor,
                "prompt": prompt_list,
                "R": R_tensor,
                "K": K_tensor,
            }
            with torch.no_grad():
                outputs = model.inference(batch)  # (1,8,res,res,3) float32 in [0,1]

            view_paths = []
            for vi in range(8):
                arr = outputs[0, vi]
                arr_uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                img_pil = Image.fromarray(arr_uint8)
                vp = os.path.join(sample_dir, f"view_{vi}.png")
                img_pil.save(vp)
                view_paths.append(vp)

            pers = [cv2.imread(p) for p in view_paths]
            angles = [[90, 45 * i, 0] for i in range(8)]
            ee = m_P2E.Perspective(pers, angles)

            pano_small = ee.GetEquirec(512, 1024)
            cv2.imwrite(os.path.join(sample_dir, "pano_512x1024.png"), pano_small)

            pano_large = ee.GetEquirec(1024, 2048)
            cv2.imwrite(os.path.join(sample_dir, "pano_1024x2048.png"), pano_large)

            print(f"Sample {si} complete, saved under '{sample_dir}'")

        print(f"All {args.batch_size} samples saved in '{out_base}'.")

    else:
        # -----------------------------------
        # Dataset‐Driven Inference/Validation
        # -----------------------------------
        image_root_dir = "training/mp3d_skybox"
        mode_split = "train" if args.eval_on_train else args.mode

        if config["dataset"]["name"] == "mp3d":
            dataset = MP3Ddataset(config["dataset"], mode=mode_split)
        elif config["dataset"]["name"] == "scannet":
            dataset = Scannetdataset(config["dataset"], mode=mode_split)
        else:
            raise ValueError(f"Unknown dataset name: {config['dataset']['name']}")

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        logger = TensorBoardLogger(
            save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False
        )

        trainer = pl.Trainer.from_argparse_args(args, logger=logger)

        if args.mode == "test":
            trainer.test(model, data_loader)
        else:
            trainer.validate(model, data_loader)
