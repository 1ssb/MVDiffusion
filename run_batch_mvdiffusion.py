import torch
import argparse
import yaml
from src.lightning_pano_gen import PanoGenerator
from src.lightning_pano_outpaint import PanoOutpaintGenerator
import numpy as np
import cv2
import os
from generate_video_tool.pano_video_generation import generate_video
from PIL import Image
import generate_video_tool.lib.multi_Perspec2Equirec as m_P2E
import time

torch.manual_seed(0)

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gen_video',
                    action='store_true', help='generate video')
    parser.add_argument('--text_path',
                    type=str, help='text path allow to specify 8 texts')
    parser.add_argument('--out_dir',
                    type=str, default='test_text2room', help='output directory')

    return parser.parse_args()

def resize_and_center_crop(img, size):
    H, W, _ = img.shape
    if H==W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size*H/W)
        img = cv2.resize(img, (size, current_size))
        # center crop to square
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[margin_l:-margin_r, :]
    else:
        current_size=int(size*W/H)
        img = cv2.resize(img, (current_size, size))
        margin_l=(current_size-size)//2
        margin_r=current_size-margin_l-size
        img=img[:, margin_l:-margin_r]
    return img

SCENE_TEXT_MAP = {
    # 'bedroom_scene_03084_641793': 'The bedroom has four walls. The room has a window and a sofa . The table is to the left of the sofa',
    # 'bedroom_scene_03113_560': 'The bedroom has six walls. The room has a cabinet and a window',
    # 'bedroom_scene_03243_800738': 'The bedroom has four walls. The room has a window and a picture',
    # 'bedroom_scene_03422_794742': 'The bedroom has four walls. The room has a cabinet and a window ',
    # 'livingroom_scene_03027_2723': 'The living room has fourteen walls. The room has a cabinet and a chair ', 
    # 'livingroom_scene_03049_284331': 'The living room has eight walls. The room has a picture , a shelves and a cabinet ', 
    # 'livingroom_scene_03125_536': 'The living room has twelve walls. The room has a cabinet and a window . There is a lamp above the cabinet ', 
    # 'livingroom_scene_03200_522357': 'The living room has six walls. The room has a cabinet , a fridge and a window . There is a chair to the left of the fridge . There is a second chair behind the first chair', 
    # 'livingroom_scene_03300_190736': 'The living room has ten walls. The room has a picture and a window ', 
    # 'livingroom_scene_03309_12023': 'The living room has ten walls. The room has a cabinet , a fridge and a shelves . The fridge is to the right of the cabinet . There is a chair to the left of the shelves ', 
    'kitchen_scene_03013_117': 'The kitchen has four walls.The room has a fridge , a window and a sink .',
    'kitchen_scene_03406_159': 'The kitchen has six walls.The room has a fridge , a window and a cabinet .The cabinet is to the left of the fridge .There is a stove to the left of the fridge .',
}

if __name__ == '__main__':
    args = parse_args()

    config_file = 'configs/pano_generation.yaml'
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    model = PanoGenerator(config)
    model.load_state_dict(torch.load('weights/pano.ckpt', map_location='cpu')[
            'state_dict'], strict=True)
    model=model.cuda()
    img=None
    
    pano_resolution=(2048, 1024)
    resolution=config['dataset']['resolution']
    Rs=[]
    Ks=[]
    for i in range(8):
        degree = (45*i) % 360
        K, R = get_K_R(FOV=90, THETA=degree,PHI= 0,
                        height=resolution, width=resolution)

        Rs.append(R)
        Ks.append(K)

    images=torch.zeros((1,8,resolution,resolution, 3)).cuda()
    if img is not None:
        images[0,0]=img

    for scene_name, text_prompt in SCENE_TEXT_MAP.items():
        print('scene: {}'.format(scene_name))
        print('text: {}'.format(text_prompt))

        begin_tms = time.time()

        prompt=[text_prompt]*8                
        K=torch.tensor(Ks).cuda()[None]
        R=torch.tensor(Rs).cuda()[None]
        batch= {
                'images': images,
                'prompt': prompt,
                'R': R,
                'K': K
            }
        images_pred=model.inference(batch)

        res_dir=os.path.join(args.out_dir, scene_name)
        print('save in fold: {}'.format(res_dir))
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, 'prompt.txt'), 'w') as f:
            f.write(text_prompt)

        image_paths=[]
        for i in range(8):
            im = Image.fromarray(images_pred[0,i])
            image_path=os.path.join(res_dir, '{}.png'.format(i))
            image_paths.append(image_path)
            im.save(image_path)
        
        pers = [cv2.imread(image_path) for image_path in image_paths]
        ee = m_P2E.Perspective(pers,
                                [[90, 0, 0], [90, 45, 0], [90, 90, 0], [90, 135, 0],
                                [90, 180, 0], [90, 225, 0], [90, 270, 0], [90, 315, 0]]
                                )
        
        new_pano = ee.GetEquirec(512, 1024)
        cv2.imwrite(os.path.join(res_dir, '_pano.png'), new_pano.astype(np.uint8))

        new_pano = ee.GetEquirec(1024, 2048)
        cv2.imwrite(os.path.join(res_dir, '_pano_sr.png'), new_pano.astype(np.uint8))
        # calc sampling time
        elaps_time = time.time() - begin_tms
        print(f'sample time: {elaps_time}')
