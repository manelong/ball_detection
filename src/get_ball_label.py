"""
predict the videos
"""
import os
import cv2
from collections import deque
from PIL import Image
import numpy as np
from dataloaders.dataset_loader import get_transform
from collections import defaultdict
from dataloaders import build_img_transforms
import torch

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from utils import mkdir_if_missing
from detectors import build_detector
from trackers import build_tracker
import json
from matplotlib import pyplot as plt
from utils.dataclasses import Center
from utils import gen_video
from utils.vis import draw_frame_multiball
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import random
from trackers.online_multiball import Sort

log = logging.getLogger(__name__)


def calculate_distance_matrix(trajectories, current_points):
    """
    计算当前点和已存在轨迹末端点之间的距离矩阵
    """
    dist_matrix = np.zeros((len(trajectories), len(current_points)))
    for i, traj in enumerate(trajectories):
        for j, point in enumerate(current_points):
            dist_matrix[i, j] = np.linalg.norm(np.array(traj[-1]) - np.array(point))
    return dist_matrix


def cluster(points):
    # DBSCAN参数
    epsilon = 3  # 定义点之间的最大距离（小于该距离的点会被聚类到一起）
    min_samples = 1  # 定义每个簇的最小点数，这里设为1表示只要有距离小于epsilon的点就算一个簇

    # 初始化DBSCAN并进行聚类
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)

    # 获取聚类标签
    labels = db.labels_

    # 合并点
    clustered_points = []
    for label in set(labels):
        if label == -1:
            continue  # 噪声点忽略
        cluster = points[labels == label]
        centroid = cluster.mean(axis=0)
        clustered_points.append(centroid)

    return clustered_points


@hydra.main(version_base=None, config_name='eval', config_path='configs')
def main(
        cfg: DictConfig
):
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg)
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])
    print("out_path:",cfg['output_dir']) #打印输出路径

    # 保存本次运行的cfg到cfg['output_dir']
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(cfg['output_dir'] + '/config.json', 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

    video_dir = cfg['video_dir']
    use_tracker = False
    video_list = os.listdir(video_dir)
    
    for video_name in video_list:
        video_path = os.path.join(video_dir, video_name)
        vis_frame_dir = os.path.join(cfg['output_dir'], video_name)
        vis_frame_dir = vis_frame_dir.replace('.mp4', '')

        input_frames = 3
        _input_wh = (512, 288)
        _output_wh = (512, 288)
        _out_scales = [0]
        _is_train = False

        # 如果路径vis_frame_dir不存在，则创建路径
        if not os.path.exists(vis_frame_dir):
            os.makedirs(vis_frame_dir)

        # build the detector and tracker
        detector = build_detector(cfg, model=None)
        tracker = build_tracker(cfg)

        transform_train, transform_test = build_img_transforms(cfg=None)

        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(7))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # a queue to store the last 3 frames
        frames_deque = deque(maxlen=input_frames)
        id_deque = deque(maxlen=input_frames)

        trans_input = None
        trans_outputs = defaultdict(list)
        trans_outputs_inv = defaultdict(list)

        det_results = defaultdict(list)
        hm_results = defaultdict(list)

        for i in range(n_frames):
            ret, img = cap.read()
            if not ret:
                break
            # 将img转换为RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 增加滤波
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # change the img from numpy to PIL.Image
            img = Image.fromarray(img)

            if trans_input is None:
                trans_input = get_transform(np.asarray(img), _input_wh)
                out_w, out_h = _output_wh
                for scale in _out_scales:
                    trans_outputs[scale] = get_transform(np.asarray(img), (out_w, out_h))
                    out_w = out_w // 2
                    out_h = out_h // 2

                if not _is_train:
                    trans_input_inv = get_transform(np.asarray(img), _input_wh, inv=1)
                    out_w, out_h = _output_wh
                    for scale in _out_scales:
                        trans_outputs_inv[scale] = torch.tensor(get_transform(np.asarray(img), (out_w, out_h), inv=1)).unsqueeze(0)
                        out_w = out_w // 2
                        out_h = out_h // 2

            img = Image.fromarray(cv2.warpAffine(np.array(img), trans_input, _input_wh, flags=cv2.INTER_LINEAR))
            img = transform_test(img)

            frames_deque.append(img)
            id_deque.append(i)
            # if the frames_deque is not full, continue
            if len(frames_deque) < input_frames:
                continue

            # turn the frames_deque into a list
            frames_list = list(frames_deque)
            imgs = torch.cat(frames_list, dim=0).unsqueeze(0)

            # predict the image
            with torch.no_grad():
                batch_results, hms_vis = detector.run_tensor(imgs, trans_outputs_inv)

                for ib in batch_results.keys():
                    for ie in batch_results[ib].keys():
                        img_path = str(id_deque[int(ie)])  # 0, 1, 2
                        preds = batch_results[ib][ie]
                        det_results[img_path].extend(preds)
                        hm_results[img_path].extend(hms_vis[ib][ie])

        result_dict = {}
        if use_tracker:
            tracker.refresh()
            for img_path, preds in det_results.items():
                result_dict[img_path] = tracker.update(preds)
            for img_path, preds in result_dict.items():
                if result_dict[img_path]['visi'] == True:
                    result_dict[img_path]['x'] = [preds['x']]
                    result_dict[img_path]['y'] = [preds['y']]
        else:
            for k in det_results.keys():
                if len(det_results[k]) > 0:
                    xys = np.array([det['xy'] for det in det_results[k]])
                    xys = cluster(xys)  # [array(x, y), ...]
                    result_dict[k] = {'x': [xy[0] for xy in xys], 'y': [xy[1] for xy in xys], 'visi': True}
                else:
                    result_dict[k] = {'x': -np.Inf, 'y': -np.Inf, 'visi': False, 'score': -np.Inf}
        
        # 根据检测点生成轨迹
        dist_threshold = 50  # 距离阈值
        trajectories = []
        trajectory_colors = []  # 用于存储每条轨迹的颜色
        cm_pred = plt.get_cmap('Reds', len(result_dict))
        cap = cv2.VideoCapture(video_path)
        colors = plt.get_cmap('hsv', 100)  # 使用较多的颜色，以提供足够的颜色选择
        min_trajectory_length = 10  # 最短轨迹长度阈值

        ### original 画完整点轨迹
        for cnt, img_path in enumerate(result_dict.keys()):
            vis_frame_path = os.path.join(vis_frame_dir, os.path.basename(img_path) + '.jpg')
            _, vis_pred = cap.read()
            # vis_pred = cv2.imread(img_path)

            for cnt2, img_path2 in enumerate(result_dict.keys()):
                if cnt2 > cnt:
                    break
                if cnt-cnt2 > 5:
                    continue
                color_pred = (int(cm_pred(cnt2)[2] * 255), int(cm_pred(cnt2)[1] * 255), int(cm_pred(cnt2)[0] * 255))
                x_pred = result_dict[img_path2]['x']  # [x1, x2, ...]
                y_pred = result_dict[img_path2]['y']  # [y1, y2, ...]
                visi_pred = result_dict[img_path2]['visi']

                vis_pred = draw_frame_multiball(vis_pred, xs=x_pred, ys=y_pred, is_visible=visi_pred,
                                                color=color_pred, radius=8)
                               
            
            # cv2.imwrite(vis_frame_path, vis_pred)

        # video_path = '{}.mp4'.format(vis_frame_dir)
        # gen_video(video_path, vis_frame_dir, fps=25.0)
        # print("video_path",video_path)



if __name__ == "__main__":
    main()
