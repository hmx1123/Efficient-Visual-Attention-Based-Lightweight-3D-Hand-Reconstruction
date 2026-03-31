import argparse
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ptflops import get_model_complexity_info

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from dataset.interhand import fix_shape, InterHand_dataset
from dataset.dataset_utils import IMG_SIZE, cut_img
from utils.utils import get_mano_path
from utils.vis_utils import mano_two_hands_renderer
from utils.config import load_cfg
from models.manolayer import ManoLayer
from models.model import load_model

class Jr():
    def __init__(self, J_regressor,
                 device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


class handDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, dense, hand_dict = self.dataset[idx]
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(
            img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        maskTensor = torch.tensor(mask, dtype=torch.float32) / 255

        joints_left_gt = torch.from_numpy(
            hand_dict['left']['joints3d']).float()
        verts_left_gt = torch.from_numpy(hand_dict['left']['verts3d']).float()
        joints_right_gt = torch.from_numpy(
            hand_dict['right']['joints3d']).float()
        verts_right_gt = torch.from_numpy(
            hand_dict['right']['verts3d']).float()

        return imgTensor, maskTensor, joints_left_gt, verts_left_gt, joints_right_gt, verts_right_gt


def calculate_pck(pred_keypoints, gt_keypoints, threshold_scale=0.05, ref_bone='bbox_diagonal'):
    """
    计算 PCK@0.05 指标

    Args:
        pred_keypoints: 预测的关键点, tensor of shape (b, 21, 3)
        gt_keypoints: 真实的关键点, tensor of shape (b, 21, 3)
        threshold_scale: 阈值比例, 默认0.05
        ref_bone: 参考尺度类型
            - 'bbox_diagonal': 使用边界框对角线 (默认)
            - 'torso': 使用躯干距离 (对于手部不适用)
            - 'palm': 使用手掌对角线 (对于手部)

    Returns:
        pck_score: PCK@0.05 分数
        per_joint_pck: 每个关节的PCK分数
    """
    batch_size = pred_keypoints.shape[0]
    num_joints = pred_keypoints.shape[1]

    # 计算每个关键点的欧氏距离误差
    distances = torch.norm(pred_keypoints - gt_keypoints,
                           dim=2)  # shape: (b, 21)

    # 计算阈值
    thresholds = []
    for i in range(batch_size):
        if ref_bone == 'bbox_diagonal':
            # 使用边界框对角线作为参考
            bbox_min = torch.min(gt_keypoints[i], dim=0)[0]  # (3,)
            bbox_max = torch.max(gt_keypoints[i], dim=0)[0]  # (3,)
            ref_length = torch.norm(bbox_max - bbox_min)

        elif ref_bone == 'palm':
            # 对于手部，使用手掌关键点形成的对角线
            # 手掌关键点索引: 0(手腕), 1(拇指根部), 5(食指根部), 9(中指根部), 13(无名指根部), 17(小指根部)
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_points = gt_keypoints[i, palm_indices]  # (6, 3)
            bbox_min = torch.min(palm_points, dim=0)[0]
            bbox_max = torch.max(palm_points, dim=0)[0]
            ref_length = torch.norm(bbox_max - bbox_min)

        else:
            raise ValueError(f"不支持的参考尺度类型: {ref_bone}")

        threshold = threshold_scale * ref_length
        thresholds.append(threshold)

    thresholds = torch.tensor(
        thresholds, device=pred_keypoints.device)  # shape: (b,)
    thresholds = thresholds.unsqueeze(
        1).expand(-1, num_joints)  # shape: (b, 21)

    # 计算正确预测的关键点
    correct_predictions = distances < thresholds  # shape: (b, 21)

    # 计算每个关节的PCK
    per_joint_pck = correct_predictions.float().mean(dim=0)  # shape: (21,)

    # 计算整体PCK
    pck_score = correct_predictions.float().mean()

    return pck_score, per_joint_pck


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str,
                        default='misc/model/interhand.pth')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--bs", type=int, default=32)
    opt = parser.parse_args()

    opt.map = False

    network = load_model(opt.cfg)

    state = torch.load(opt.model, map_location='cpu')
    # try:
    network.load_state_dict(state, strict=True)
    # except:
    #     state2 = {}
    #     for k, v in state.items():
    #         state2[k[8:]] = v
    #     network.load_state_dict(state2)

    network.eval()
    network.cuda()

    mano_path = get_mano_path()
    mano_layer = {'left': ManoLayer(mano_path['left'], center_idx=None),
                  'right': ManoLayer(mano_path['right'], center_idx=None)}
    fix_shape(mano_layer)
    J_regressor = {'left': Jr(mano_layer['left'].J_regressor),
                   'right': Jr(mano_layer['right'].J_regressor)}

    faces_left = mano_layer['left'].get_faces()
    faces_right = mano_layer['right'].get_faces()

    dataset = handDataset(InterHand_dataset(opt.data_path, split='test'))
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)

    joints_loss = {'left': [], 'right': []}
    verts_loss = {'left': [], 'right': []}
    verts_pck = {'left': [], 'right': []}

    with torch.no_grad():
        for data in tqdm(dataloader):
            imgTensors = data[0].cuda()
            joints_left_gt = data[2].cuda()
            verts_left_gt = data[3].cuda()
            joints_right_gt = data[4].cuda()
            verts_right_gt = data[5].cuda()

            joints_left_gt = J_regressor['left'](verts_left_gt)
            joints_right_gt = J_regressor['right'](verts_right_gt)

            root_left_gt = joints_left_gt[:, 9:10]
            root_right_gt = joints_right_gt[:, 9:10]
            length_left_gt = torch.linalg.norm(
                joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
            length_right_gt = torch.linalg.norm(
                joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
            joints_left_gt = joints_left_gt - root_left_gt
            verts_left_gt = verts_left_gt - root_left_gt
            joints_right_gt = joints_right_gt - root_right_gt
            verts_right_gt = verts_right_gt - root_right_gt

            result, paramsDict, handDictList, otherInfo = network(imgTensors)

            verts_left_pred = result['verts3d']['left']
            verts_right_pred = result['verts3d']['right']
            joints_left_pred = J_regressor['left'](verts_left_pred)
            joints_right_pred = J_regressor['right'](verts_right_pred)

            root_left_pred = joints_left_pred[:, 9:10]
            root_right_pred = joints_right_pred[:, 9:10]
            length_left_pred = torch.linalg.norm(
                joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
            length_right_pred = torch.linalg.norm(
                joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
            scale_left = (length_left_gt /
                          length_left_pred).unsqueeze(-1).unsqueeze(-1)
            scale_right = (length_right_gt /
                           length_right_pred).unsqueeze(-1).unsqueeze(-1)

            joints_left_pred = (joints_left_pred - root_left_pred) * scale_left
            verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
            joints_right_pred = (joints_right_pred -
                                 root_right_pred) * scale_right
            verts_right_pred = (verts_right_pred -
                                root_right_pred) * scale_right

            verts_left_pck, _ = calculate_pck(
                joints_left_pred, joints_left_gt, threshold_scale=0.05, ref_bone='palm')
            verts_pck['left'].append(verts_left_pck.cpu().numpy())
            verts_right_pck, _ = calculate_pck(
                joints_right_pred, joints_right_gt, threshold_scale=0.05, ref_bone='palm')
            verts_pck['right'].append(verts_right_pck.cpu().numpy())

            joint_left_loss = torch.linalg.norm(
                (joints_left_pred - joints_left_gt), ord=2, dim=-1)
            joint_left_loss = joint_left_loss.detach().cpu().numpy()
            joints_loss['left'].append(joint_left_loss)

            joint_right_loss = torch.linalg.norm(
                (joints_right_pred - joints_right_gt), ord=2, dim=-1)
            joint_right_loss = joint_right_loss.detach().cpu().numpy()
            joints_loss['right'].append(joint_right_loss)

            vert_left_loss = torch.linalg.norm(
                (verts_left_pred - verts_left_gt), ord=2, dim=-1)
            vert_left_loss = vert_left_loss.detach().cpu().numpy()
            verts_loss['left'].append(vert_left_loss)

            vert_right_loss = torch.linalg.norm(
                (verts_right_pred - verts_right_gt), ord=2, dim=-1)
            vert_right_loss = vert_right_loss.detach().cpu().numpy()
            verts_loss['right'].append(vert_right_loss)

    joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
    joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
    verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
    verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)

    joints_mean_loss_left = joints_loss['left'].mean()*1000
    joints_mean_loss_right = joints_loss['right'].mean()*1000
    verts_mean_loss_left = verts_loss['left'].mean()*1000
    verts_mean_loss_right = verts_loss['right'].mean()*1000
    verts_pck_left = np.mean(verts_pck['left'])
    verts_pck_right  = np.mean(verts_pck['right'])
    
    flops, params = get_model_complexity_info(network, (imgTensors.shape[1], imgTensors.shape[2], imgTensors.shape[3]),
                                              as_strings=True, print_per_layer_stat=False)
    print(f"model:{opt.model}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    print('joint mean error:')
    print('    left: {} mm'.format(joints_mean_loss_left))
    print('    right: {} mm'.format(joints_mean_loss_right))
    print('    all: {} mm'.format(
        (joints_mean_loss_left + joints_mean_loss_right) / 2))
    print('vert mean error:')
    print('    left: {} mm'.format(verts_mean_loss_left))
    print('    right: {} mm'.format(verts_mean_loss_right))
    print('    all: {} mm'.format(
        (verts_mean_loss_left + verts_mean_loss_right) / 2))
    print('vert pck@0.05:')
    print('    left: {} %'.format(verts_pck_left))
    print('    right: {} %'.format(verts_pck_right))
    print('    all: {} %'.format(
        (verts_pck_left + verts_pck_right) / 2))
