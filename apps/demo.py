import numpy as np
import torch
import cv2 as cv
import glob
import os
import argparse
import albumentations as A
    
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from core.test_utils import InterRender
from dataset.dataset_utils import IMG_SIZE
from utils.utils import get_mano_path, imgUtils
from utils.config import load_cfg
from models.model import load_model

def cut_img(img, bbox):
    cut = img[max(int(bbox[2]), 0):min(int(bbox[3]), img.shape[0]),
              max(int(bbox[0]), 0):min(int(bbox[1]), img.shape[1])]
    cut = cv.copyMakeBorder(cut,
                            max(int(-bbox[2]), 0),
                            max(int(bbox[3] - img.shape[0]), 0),
                            max(int(-bbox[0]), 0),
                            max(int(bbox[1] - img.shape[1]), 0),
                            borderType=cv.BORDER_CONSTANT,
                            value=(0, 0, 0))
    return cut

def combine_img(bg_img):
    c,h, w = bg_img.shape
    bg_img = bg_img.reshape(-1,3,h,w)
    bg_img=bg_img.sum(dim=0)
    return bg_img
    

def reverse_processing_dense(dense_map_tensor, original_size):
    """
    反向处理过程
    Args:
        dense_map_tensor: 经过处理的张量，形状为 (C, H, W)
        original_size: 原始图像尺寸 (width, height) 或 (height, width)
    
    Returns:
        original_dense_map: 恢复后的原始密集图
    """
    dense_map_tensor=combine_img(dense_map_tensor)
    # 步骤1: 反向维度变换
    dense_map = dense_map_tensor.cpu().permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
    
    # 步骤2: 反归一化
    dense_map = (dense_map * 255).type(torch.uint8)
    
    # 步骤3: 转换为numpy
    dense_map_np = dense_map.numpy()
    
    # 步骤4: 反向resize
    # 确保original_size格式为 (width, height)
    if len(original_size) == 2:
        if original_size[0] > original_size[1]:  # 假设宽大于高
            target_size = (original_size[0], original_size[1])
        else:
            target_size = (original_size[1], original_size[0])
    else:
        target_size = original_size
        
    original_dense_map = cv.resize(dense_map_np, target_size)
    
    return original_dense_map

def reverse_processing_mask(mask_tensor, original_size, flip=False, first_channel_value=0):
    """
    更精确的反向处理mask张量
    
    Args:
        mask_tensor: 处理后的mask张量，形状为 (C, H, W)
        original_size: 原始图像尺寸 (width, height)
        flip: 是否在原始处理中进行了通道翻转
        first_channel_value: 被删除的第一个通道的填充值 (0或1)
    
    Returns:
        original_mask: 恢复后的原始mask
    """
    # 1. 反向维度变换
    mask = mask_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 2. 反向通道翻转
    if flip:
        mask = mask[..., [1, 0]]
    
    # 3. 恢复第一个通道
    h, w, c = mask.shape
    if first_channel_value == 0:
        first_channel = np.zeros((h, w, 1))
    else:
        first_channel = np.ones((h, w, 1))
    
    mask_3ch = np.concatenate([first_channel, mask], axis=2)
    
    # 4. 反归一化
    mask_uint8 = (mask_3ch * 255).astype(np.uint8)
    
    # 5. 反向resize（使用最近邻插值保持二值特性）
    original_mask = cv.resize(mask_uint8, original_size, interpolation=cv.INTER_NEAREST)
    
    return original_mask

def reverse_processing_hms(hms_tensor, original_sizes, flip=False, num_keypoints=21):
    """
    通用的反向处理热图张量
    
    Args:
        hms_tensor: 处理后的热图张量，形状为 (C, H, W)
        original_sizes: 每个热图的原始尺寸列表 [(w1, h1), (w2, h2), ...]
        flip: 是否在原始处理中进行了通道翻转
        num_keypoints: 每个热图的关键点数量
    
    Returns:
        original_hms: 恢复后的原始热图列表
    """
    hms_tensor=combine_img(hms_tensor)
    # 1. 反向维度变换
    hms = hms_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 2. 反归一化
    hms = (hms * 255).astype(np.uint8)
    
    # 3. 反向通道翻转
    if flip:
        # 创建反向索引
        num_total_keypoints = hms.shape[-1]
        num_hms = num_total_keypoints // num_keypoints
        reverse_idx = []
        for i in range(num_hms):
            start = i * num_keypoints
            end = (i + 1) * num_keypoints
            # 对每个热图的关键点进行反向翻转
            reverse_idx.extend(range(start + num_keypoints // 2, end))
            reverse_idx.extend(range(start, start + num_keypoints // 2))
        hms = hms[..., reverse_idx]
    
    # 4. 拆分热图
    num_hms = len(original_sizes)
    channels_per_hm = hms.shape[-1] // num_hms
    
    original_hms = []
    for i, original_size in enumerate(original_sizes):
        start_idx = i * channels_per_hm
        end_idx = (i + 1) * channels_per_hm
        hm = hms[..., start_idx:end_idx]
        
        # 5. 反向resize到原始尺寸
        original_hm = cv.resize(hm, original_size, interpolation=cv.INTER_LINEAR)
        original_hms.append(original_hm)
    
    return original_hms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str,
                        default='misc/model/wild_demo.pth')
    parser.add_argument("--live_demo", action='store_true')
    parser.add_argument("--img_path", type=str, default='demo/')
    parser.add_argument("--save_path", type=str, default='demo/')
    parser.add_argument("--render_size", type=int, default=256)
    opt = parser.parse_args()

    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=opt.render_size)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not opt.live_demo:
        # img_path_list = glob.glob(os.path.join(
        #     opt.img_path, '*.jpg')) + glob.glob(os.path.join(opt.img_path, '*.png'))
        extensions = ['*.jpg', '*.JPG', '*.jpeg',
                      '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']
        img_path_list = []
        for ext in extensions:
            img_path_list.extend(glob.glob(os.path.join(opt.img_path, ext)))
        img_path_list.sort()
        targets = np.arange(500, 25001, 2000).tolist()
        
        def invert_image(img, **kwargs):
            return 255 - img
        transform = A.Compose([
            # A.CoarseDropout(num_holes=4, max_h_size=8, max_w_size=8, fill_value=(250, 210, 190), p=1),
            A.RandomBrightnessContrast(p=0.5),
            # A.HueSaturationValue(
            #     hue_shift_limit=20,  # 色调偏移限制
            #     sat_shift_limit=30,  # 饱和度偏移限制
            #     val_shift_limit=20,  # 明度偏移限制
            #     p=0.7
            # ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20,
                       b_shift_limit=20, p=0.5),
            A.ChannelShuffle(p=0.2),
            A.RandomGamma(p=0.3),
            A.Lambda(image=invert_image, p=0.3),  # 亮度反转
            # A.Blur(blur_limit=101, p=1),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2(),
        ])

        for img_path in img_path_list:
            img_name = os.path.basename(img_path)
            if img_name.find('output.jpg') != -1:
                continue
            img_name = img_name[:img_name.find('.')]
            # if int(img_name) % 5000 == 0:  # 指定文件
            if any(str(num) in img_name for num in targets):
                img = cv.imread(img_path)
                img = imgUtils.pad2squre(img)
                img = cv.resize(img, (256, 256))
                params, result, paramsDict, handDictList, otherInfo = model.run_model(
                    img)
                img_overlap = model.render(params, bg_img=img)
                # cv.imwrite(os.path.join(opt.save_path,
                #                         img_name + '.jpg'), img)
                # cv.imwrite(os.path.join(opt.save_path,
                #                         img_name + '_output.jpg'), img_overlap)
                if img.shape[0] != img_overlap.shape[0]:
                    # 计算缩放比例，保持宽高比
                    scale = img.shape[0] / img_overlap.shape[0]
                    new_width = int(img_overlap.shape[1] * scale)
                    img_overlap = cv.resize(
                        img_overlap, (new_width, img.shape[0]))
                img_tran=transform(image=img)
                img_tran=img_tran['image']
                
                mask=otherInfo['mask']
                mask=reverse_processing_mask(mask.squeeze(),(256,256),flip=False,first_channel_value=0)
                hms=otherInfo['hms']
                hms=reverse_processing_hms_list=reverse_processing_hms(hms.squeeze(),[(256,256)],flip=False,num_keypoints=21)
                dense=otherInfo['dense']
                dense=reverse_processing_dense(dense.squeeze(),(256,256))
                
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_ori_img.jpg'), img)
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_output_img.jpg'), img_overlap)
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_trans_img.jpg'), img_tran)
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_mask.jpg'), mask)
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_hms.jpg'), hms[0])
                cv.imwrite(os.path.join(opt.save_path,
                                        img_name + '_dense.jpg'), dense)


    else:
        video_reader = cv.VideoCapture(0)
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_reader.set(cv.CAP_PROP_FOURCC, fourcc)

        smooth = False
        params_last = None
        params_last_v = None
        params_v = None
        params_a = None

        fIdx = 0
        with torch.no_grad():
            while True:
                fIdx = fIdx + 1
                _, img = video_reader.read()
                if img is None:
                    exit()
                w = min(img.shape[1], img.shape[0]) / 2 * 0.6
                left = int(img.shape[1] / 2 - w)
                top = int(img.shape[0] / 2 - w)
                size = int(2 * w)
                bbox = [left, left + size, top, top + size]
                bbox = np.array(bbox).astype(np.int32)
                crop_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                params = model.run_model(crop_img)
                if smooth and params_last is not None and params_v is not None and params_a is not None:
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            pred = params_last[k] + \
                                params_v[k] + 0.5 * params_a[k]
                            params[k] = (0.7 * params[k] + 0.3 * pred)

                img_out = model.render(params, bg_img=crop_img)
                img[bbox[2]:bbox[3], bbox[0]:bbox[1]
                    ] = cv.resize(img_out, (size, size))
                cv.line(img, (int(bbox[0]), int(bbox[2])), (int(
                    bbox[0]), int(bbox[3])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[1]), int(bbox[2])), (int(
                    bbox[1]), int(bbox[3])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[0]), int(bbox[2])), (int(
                    bbox[1]), int(bbox[2])), (0, 0, 255), 2)
                cv.line(img, (int(bbox[0]), int(bbox[3])), (int(
                    bbox[1]), int(bbox[3])), (0, 0, 255), 2)
                cv.imshow('cap', img)

                if params_last is not None:
                    params_v = {}
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            params_v[k] = (params[k] - params_last[k])
                if params_last_v is not None and params_v is not None:
                    params_a = {}
                    for k in params.keys():
                        if isinstance(params[k], torch.Tensor):
                            params_a[k] = (params_v[k] - params_last_v[k])
                params_last = params
                params_last_v = params_v

                key = cv.waitKey(1)

                if key == 27:
                    exit()
