from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio
import torch
import argparse
import tqdm
import os
import cv2
import numpy as np
from einops import rearrange
from src.dataset.utils import get_K_R
from src.dataset.Matterport3D import warp_img
from src.models.pano.utils import get_correspondences
from collections import defaultdict
import torch.nn.functional as F

torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--result_dir', type=str, required=True,
    #                     help='Directory where results are saved')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')

    return parser.parse_args()


def compute_psnr_masked(img1, img2, mask):
    img1_masked = img1[mask]/255
    img2_masked = img2[mask]/255

    mse = np.mean((img1_masked - img2_masked)**2)

    if mse == 0:
        return float('inf')

    max_pixel_value = 255.0
    psnr = -10*np.log10(mse)

    return psnr


def compute_psnr(img1, img2, K, R):
    im_h, im_w, _ = img1.shape
    homo_matrix = K@R@np.linalg.inv(K)
    mask = np.ones((im_h, im_w))
    img_warp2 = cv2.warpPerspective(img2, homo_matrix, (im_w, im_h))
    mask = cv2.warpPerspective(mask, homo_matrix, (im_w, im_h)) == 1
    psnr = compute_psnr_masked(img1, img_warp2, mask)
    
    return psnr


class MVResultDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.gen_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/MVDiffusion/finetune_mvd_8'
        self.gt_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
        self.prompt_path = '/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/test/'
        self.room_list = ['bedroom', 'livingroom']
        # self.room_list = ['bedroom']
        self.scene_path_list, self.scene_name = [], []
        for room in self.room_list:
            room_folderpath = os.path.join(self.gen_path, room)
            scene_list = sorted(os.listdir(room_folderpath))
            for scene_name in scene_list:
                scene_folderpath = os.path.join(room_folderpath, scene_name, 'mvd_img')
                self.scene_name.append(scene_name)
                self.scene_path_list.append(scene_folderpath)

    def __len__(self):
        return len(self.scene_path_list)

    def __getitem__(self, idx):
        num_views = 8
        images_gt = []
        images_gen = []
        Rs = []
        cameras = defaultdict(list)
        appendix = ['_0.png', '_45.png', '_90.png', '_135.png', '_180.png', '_225.png', '_270.png', '_315.png']
        prompt = []
        for i in range(num_views):
            # for images, ext in zip([images_gt, images_gen], ["_natural.png", ".png"]):
            #     img = cv2.imread(os.path.join(self.result_dir, self.scenes[idx], f"{i}{ext}"))
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     images.append(img)
            img = cv2.imread(os.path.join(self.scene_path_list[idx], self.scene_name[idx]+appendix[i]))
            images_gen.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            room = self.scene_path_list[idx].split('/')[-3]
            img_gt = cv2.imread(os.path.join(self.gt_path, room, "mvd_img", self.scene_name[idx]+appendix[i]))
            images_gt.append(img_gt)
            
            text = ""
            with open(os.path.join(self.prompt_path, room, "text_desc", self.scene_name[idx]+".txt"), 'r') as f:
                for j, line in enumerate(f):
                    text = text + line.strip()
            prompt=[text]*8

            theta = (360 / num_views * i) % 360
            K, R = get_K_R(90, theta, 0, *img.shape[:2])

            Rs.append(R)
            cameras['height'].append(img.shape[0])
            cameras['width'].append(img.shape[1])
            cameras['FoV'].append(90)
            cameras['theta'].append(theta)
            cameras['phi'].append(0)

        images_gt = np.stack(images_gt, axis=0)
        images_gen = np.stack(images_gen, axis=0)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)
        for k, v in cameras.items():
            cameras[k] = np.stack(v)

        # prompt_dir = os.path.join(self.result_dir, self.scenes[idx], "prompt.txt")
        # prompt = []
        # with open(prompt_dir, 'r') as f:
        #     for line in f:
        #         prompt.append(line.strip())

        return {
            'images_gt': images_gt,
            'images_gen': images_gen,
            'K': K,
            'R': R,
            'cameras': cameras,
            'prompt': prompt
        }

import glob
class OurResultDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.gen_path = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/sample_results/pano_gen_experiments/'
        self.multiviews_gt_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
        self.multiviews_prompt_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
        self.prompt_path = '/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/test/'
        self.room_list = ['bedroom', 'livingroom']
        # self.room_list = ['bedroom']
        self.scene_path_list, self.scene_name = [], []
        for room in self.room_list:
            room_folderpath = os.path.join(self.gen_path, room)
            gen_scene_list = sorted([f for f in os.listdir(room_folderpath) if os.path.isdir(os.path.join(room_folderpath, f)) and f.isdigit()], key=lambda x: int(x))
            # gen_scene_list = gen_scene_list[:50]
            for scene_name in gen_scene_list:
                scene_folderpath = os.path.join(room_folderpath, scene_name, 'mvd_img')
                # print(f"json file path: {os.path.join(room_folderpath, scene_name, '*.json')}")
                scene_json_filepath = glob.glob(os.path.join(room_folderpath, scene_name, '*.json'))[0]
                scene_name = os.path.basename(scene_json_filepath).split('.')[0]
                self.scene_name.append(scene_name)
                self.scene_path_list.append(scene_folderpath)
        
        self.use_our_text_prompt = True

    def __len__(self):
        return len(self.scene_path_list)

    def __getitem__(self, idx):
        num_views = 8
        images_gt = []
        images_gen = []
        Rs = []
        cameras = defaultdict(list)
        appendix = ['_0.png', '_45.png', '_90.png', '_135.png', '_180.png', '_225.png', '_270.png', '_315.png']
        prompt = []
        for i in range(num_views):
            subview_img_path = os.path.join(self.scene_path_list[idx], self.scene_name[idx]+"_sem"+appendix[i])
            # print(f'generated sunview image path: {subview_img_path}')
            img = cv2.imread(subview_img_path)
            images_gen.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            room = self.scene_path_list[idx].split('/')[-3]
            gt_subview_img_path = os.path.join(self.multiviews_gt_path, room, "mvd_img", self.scene_name[idx]+appendix[i])
            # print(f'ground truth sunview image path: {gt_subview_img_path}')
            img_gt = cv2.imread(gt_subview_img_path, -1)
            images_gt.append(img_gt)
            
            if self.use_our_text_prompt:
                text = ""
                with open(os.path.join(self.prompt_path, room, "text_desc", self.scene_name[idx]+".txt"), 'r') as f:
                    for j, line in enumerate(f):
                        text = text + line.strip()
                # print(f'text prompt: {text}')
                prompt=[text]*8
            else:
                prompt_path = os.path.join(self.multiviews_prompt_path, room, "mvd_text", self.scene_name[idx]+appendix[i].replace('png','txt'))
                # print(f'mvd text prompt path: {prompt_path}')
                with open(prompt_path, 'r') as f:
                    for j, line in enumerate(f):
                        text = line.strip()
                prompt.append(text)

            theta = (360 / num_views * i) % 360
            K, R = get_K_R(90, theta, 0, *img.shape[:2])

            Rs.append(R)
            cameras['height'].append(img.shape[0])
            cameras['width'].append(img.shape[1])
            cameras['FoV'].append(90)
            cameras['theta'].append(theta)
            cameras['phi'].append(0)

        images_gt = np.stack(images_gt, axis=0)
        images_gen = np.stack(images_gen, axis=0)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)
        for k, v in cameras.items():
            cameras[k] = np.stack(v)

        return {
            'images_gt': images_gt,
            'images_gen': images_gen,
            'K': K,
            'R': R,
            'cameras': cameras,
            'prompt': prompt
        }


import glob
class Text2lightResultDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.gen_path = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/HolisticDiffuScene/sample_results/pano_gen_experiments/'
        self.multiviews_gt_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
        self.multiviews_prompt_path = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
        self.prompt_path = '/mnt/nas_3dv/hdd1/datasets/datasets/Structured3D/text2pano/test/'
        self.room_list = ['bedroom', 'livingroom']
        # self.room_list = ['bedroom']
        self.scene_path_list, self.scene_name = [], []
        for room in self.room_list:
            room_folderpath = os.path.join(self.gen_path, room)
            gen_scene_list = sorted([f for f in os.listdir(room_folderpath) if os.path.isdir(os.path.join(room_folderpath, f)) and f.isdigit()], key=lambda x: int(x))
            # gen_scene_list = gen_scene_list[:50]
            for scene_name in gen_scene_list:
                scene_folderpath = os.path.join(room_folderpath, scene_name, 'text2light', 'mvd_img')
                if not os.path.exists(scene_folderpath):
                    print(f'not exist: {scene_folderpath}')
                    continue
                # print(f"json file path: {os.path.join(room_folderpath, scene_name, '*.json')}")
                scene_json_filepath = glob.glob(os.path.join(room_folderpath, scene_name, '*.json'))[0]
                scene_name = os.path.basename(scene_json_filepath).split('.')[0]
                self.scene_name.append(scene_name)
                self.scene_path_list.append(scene_folderpath)
        
        self.use_our_text_prompt = True

    def __len__(self):
        return len(self.scene_path_list)

    def __getitem__(self, idx):
        num_views = 8
        images_gt = []
        images_gen = []
        Rs = []
        cameras = defaultdict(list)
        appendix = ['_0.png', '_45.png', '_90.png', '_135.png', '_180.png', '_225.png', '_270.png', '_315.png']
        prompt = []
        for i in range(num_views):
            subview_img_path = os.path.join(self.scene_path_list[idx], 'ldr_text2light'+appendix[i])
            # print(f'generated sunview image path: {subview_img_path}')
            img = cv2.imread(subview_img_path, -1)
            # images_gen.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            images_gen.append(img)
            
            room = self.scene_path_list[idx].split('/')[-4]
            # print(f'room: {room}')
            gt_subview_img_path = os.path.join(self.multiviews_gt_path, room, "mvd_img", self.scene_name[idx]+appendix[i])
            # print(f'ground truth sunview image path: {gt_subview_img_path}')
            img_gt = cv2.imread(gt_subview_img_path, -1)
            images_gt.append(img_gt)
            
            if self.use_our_text_prompt:
                text = ""
                with open(os.path.join(self.prompt_path, room, "text_desc", self.scene_name[idx]+".txt"), 'r') as f:
                    for j, line in enumerate(f):
                        text = text + line.strip()
                # print(f'text prompt: {text}')
                prompt=[text]*8
            else:
                prompt_path = os.path.join(self.multiviews_prompt_path, room, "mvd_text", self.scene_name[idx]+appendix[i].replace('png','txt'))
                # print(f'mvd text prompt path: {prompt_path}')
                with open(prompt_path, 'r') as f:
                    for j, line in enumerate(f):
                        text = line.strip()
                prompt.append(text)

            theta = (360 / num_views * i) % 360
            K, R = get_K_R(90, theta, 0, *img.shape[:2])

            Rs.append(R)
            cameras['height'].append(img.shape[0])
            cameras['width'].append(img.shape[1])
            cameras['FoV'].append(90)
            cameras['theta'].append(theta)
            cameras['phi'].append(0)

        images_gt = np.stack(images_gt, axis=0)
        images_gen = np.stack(images_gen, axis=0)
        K = np.stack([K]*len(Rs)).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)
        for k, v in cameras.items():
            cameras[k] = np.stack(v)

        return {
            'images_gt': images_gt,
            'images_gen': images_gen,
            'K': K,
            'R': R,
            'cameras': cameras,
            'prompt': prompt
        }


if __name__ == '__main__':
    args = parse_args()

    fid = FrechetInceptionDistance(feature=2048).cuda()
    inception = InceptionScore().cuda()
    cs = CLIPScore().cuda()
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    psnrs=[]

    mvd_dataset = MVResultDataset()
    our_dataset = OurResultDataset()
    text2light_dataset = Text2lightResultDataset()
    data_loader = torch.utils.data.DataLoader(
        mvd_dataset, num_workers=args.num_workers, batch_size=args.batch_size)

    for batch in tqdm.tqdm(data_loader):
        images_gt = rearrange(batch['images_gt'].cuda(), 'b l h w c -> (b l) c h w')
        images_gen = rearrange(batch['images_gen'].cuda(), 'b l h w c -> (b l) c h w')
        fid.update(images_gt, real=True)
        fid.update(images_gen, real=False)
        inception.update(images_gen)

        prompt_reshape = sum(map(list, zip(*batch['prompt'])), [])
        cs.update(images_gen, prompt_reshape)


    print(f"FID: {fid.compute()}")
    print(f"IS: {inception.compute()}")
    print(f"CS: {cs.compute()}")
    # from cleanfid import fid
    # fdir1 = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/MVDiffusion/finetune_mvd_8'
    # fdir2 = '/mnt/nas_3dv/hdd1/datasets/huxiaotao/mvdiffusion_text2pano/test'
    # score = fid.compute_fid(fdir1, fdir2)
    # print(score)
    
    # print(f"PSNR: {psnr.compute()}")
    # print(f"PSNR (author's code): {np.mean(psnrs)}")