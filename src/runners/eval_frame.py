"""
输出是test_npz文件，据其计算当前模型的指标结果
"""
from .base import BaseRunner
from omegaconf import DictConfig
import torch
from dataloaders import build_dataloader,build_dataloader_multiball
from models import build_model
import os, logging
from utils import count_params, Evaluator, set_seed
from detectors import build_detector
from trackers import build_tracker
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

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



def filter_valid_entries(xy_visis_tuple):
    """
        过滤输入中的零元素
    """
    xy_list, visi_list = xy_visis_tuple
    
    # 判断是否为全 (0, 0)
    if all(torch.all(xy == torch.tensor([0., 0.])) for xy in xy_list):
        return [torch.tensor([0., 0.])], [torch.tensor(False)]
    
    # Filter out invalid entries using list comprehension
    filtered_xy_list = []
    filtered_visi_list = []
    for xy, visi in zip(xy_list, visi_list):
        if not torch.all(xy == torch.tensor([0., 0.])):
            filtered_xy_list.append(xy)
            filtered_visi_list.append(visi)
    
    return filtered_xy_list, filtered_visi_list
    


class EvalFrameRunner(BaseRunner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        seed = self._cfg['seed']
        set_seed(seed)

        self._device = cfg['runner']['device']
        if self._device!='cuda':
            assert 0, 'device=cpu not supported'
        if not torch.cuda.is_available():
            assert 0, 'GPU NOT available'

        # dataloader
        _,self._test_loader = build_dataloader_multiball(cfg)

        # model
        self._model = build_model(cfg)
        self._model = self._model.to(self._device)
        self._model.eval()
        # log.info(self._model)
        log.info('# model params: (trainable) {}, (whole) {}'.format(count_params(self._model),
                                                                     count_params(self._model, only_trainable=False)))

        pretrained_model = cfg['runner']['pretrained_model']
        if pretrained_model and os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            print('Load pretrained param %s done!' % cfg['runner']['pretrained_model'])

        # evaluator
        self.evaluator = Evaluator(self._cfg)
        self.detector = build_detector(self._cfg, model=self._model)
        self.tracker = build_tracker(self._cfg)

    @torch.no_grad()
    def run(self):
        det_results = defaultdict(list)
        hm_results = defaultdict(list)

        xy_visis_dict = {}
        for batch_idx, (imgs, hms, trans, xys_gt, visis_gt, img_paths) in enumerate(tqdm(self._test_loader, desc='[INFERENCE]' )):
        # if batch_idx >= 10:
        #     break          
            batch_results, hms_vis = self.detector.run_tensor(imgs, trans)
            img_paths = [list(in_tuple) for in_tuple in img_paths]

            # xys_gt [N, 3, 2]; visis_gt [N, 3]; img_paths [3, N]
            for i in range(xys_gt.size()[0]):  # batch size
                for j in range(xys_gt.size()[1]):  # step size
                    img_path = img_paths[j][i]
                    gt = xys_gt[i][j]
                    visis = visis_gt[i][j]
                    if img_path not in list(xy_visis_dict.keys()):
                        xy_visis_dict[img_path] = [gt, visis]

            for ib in batch_results.keys():
                for ie in batch_results[ib].keys():
                    img_path = img_paths[ie][ib]
                    preds = batch_results[ib][ie]
                    det_results[img_path].extend(preds)
                    hm_results[img_path].extend(hms_vis[ib][ie])
  

        print('detection done!')
        '''
        self.tracker.refresh()
        result_dict = {}

        for img_path, preds in det_results.items():
            frame_results = self.tracker.update(preds)
            result_dict[img_path] = frame_results

        print('Tracking done!')
        for img_path in result_dict.keys():
            for pred in result_dict[img_path]:
                xy_pred = (pred['x'], pred['y'])
                visi_pred = pred['visi']
                # score_pred = pred['score']
                
                filtered_xy, filtered_vis = filter_valid_entries((xy_visis_dict[img_path][0], xy_visis_dict[img_path][1]))
            
                self.evaluator.eval_single_frame(xy_pred, visi_pred,
                                             filtered_xy, filtered_vis)

        print('tracking done!')
        '''

        result_dict = {}
        for k in det_results.keys():
            if len(det_results[k]) > 0:
                xys = np.array([det['xy'] for det in det_results[k]])
                xys = cluster(xys)  # [array(x, y), ...]
                result_dict[k] = {'x': [xy[0] for xy in xys], 'y': [xy[1] for xy in xys],'visi': [True] * len(xys)}
            else:
                result_dict[k] = {'x': [0], 'y': [0], 'visi': [False]}
        

        
        for cnt, img_path in enumerate(result_dict.keys()):
            xy_pred = list(zip(result_dict[img_path]['x'], result_dict[img_path]['y']))
            # x_pred = result_dict[img_path]['x']
            # y_pred = result_dict[img_path]['y']
            visi_pred = list(result_dict[img_path]['visi'])
            # score_pred = result_dict[img_path]['score']


            filtered_xy, filtered_vis = filter_valid_entries((xy_visis_dict[img_path][0], xy_visis_dict[img_path][1]))
            
            self.evaluator.eval_single_frame(xy_pred, visi_pred,
                                             filtered_xy, filtered_vis)
 

        print('evaluation done!')

        self.evaluator.print_results(
            txt='{} @ dist_threshold={}'.format(self._cfg['model']['name'], self.evaluator.dist_threshold),
            elapsed_time=1,
            num_frames=1,
            with_ap=False)
