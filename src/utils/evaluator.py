import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

# 初始化日志模块
log = logging.getLogger(__name__)

def cluster_predictions(xy_preds, eps=10, min_samples=1):
    # 聚点
    if len(xy_preds) == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_preds)
    labels = clustering.labels_

    cluster_centers = []
    for label in set(labels):
        cluster_points = xy_preds[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    
    return np.array(cluster_centers)



class Evaluator(object):
    def __init__(self, cfg):
        self._dist_threshold = cfg['runner']['eval']['dist_threshold']  # 设置距离阈值从配置文件读取
        self.reset_counts()  # 初始化统计计数器
        
    def reset_counts(self):  
        self._tp = 0           # 真正例数（True Positives）
        self._fp1 = 0         # 第一类假正例数（False Positives Type I）
        self._fp2 = 0         # 第二类假正例数（False Positives Type II）
        self._tn = 0          # 真负例数（True Negatives）
        self._fn = 0          # 假负例数（False Negatives）
        self._ses = []       
        self._scores = []    
        self._ys = []        

    def eval_single_frame(self, xy_preds, visi_preds, xy_gts, visi_gts):
        # Ensure input data can be iterated even when it contains a single element
        if not isinstance(xy_preds, list):
            xy_preds = [xy_preds]
            visi_preds = [visi_preds]
            # scores_pred = [scores_pred]

        # 转换成 NumPy 矩阵 用于计算距离
        xy_preds_arr = np.array([np.array(pred) if isinstance(pred, tuple) else pred.numpy() for pred in xy_preds])
        xy_gts_arr = np.array([gt.numpy() for gt in xy_gts])

        if len(xy_preds_arr)>1 :
            xy_preds_arr = cluster_predictions(xy_preds_arr, eps=15)  # 聚类一次

        # 初始化当前数量
        tp, fp1, fn = 0, 0, 0
        ses = []
        matched_gts = set()
        matched_preds = set()

        # 计算距离矩阵
        dist_matrix = cdist(xy_preds_arr, xy_gts_arr)

        # 得到预测点到标签点的最小距离值和位置
        for pred_idx, (pred_pos, visi_pred) in enumerate(zip(xy_preds_arr, visi_preds)):
            min_dist = float('inf')
            closest_gt_idx = None
            
            for gt_idx, (gt_pos, visi_gt) in enumerate(zip(xy_gts_arr, visi_gts)):
                dist = dist_matrix[pred_idx, gt_idx]
                if dist < min_dist:
                    min_dist = dist
                    closest_gt_idx = gt_idx

            if closest_gt_idx is not None and min_dist < self._dist_threshold:
                tp += 1  # 距离小于阈值，正确检测
                ses.append(min_dist ** 2)
                matched_gts.add(closest_gt_idx)
            else:
                fp1 += 1  # 距离大于阈值



        fn=len(xy_gts_arr) - len(matched_gts)  #所有真实点中没有被任何预测点匹配的点

        # Update total counts
        self._tp += tp
        self._fp1 += fp1
        self._fn += fn

        if tp > 0 or fp1 > 0:
            self._ys.extend([int(gt_idx in matched_gts) for gt_idx in range(len(visi_gts))])

        return {'tp': tp, 'fp1': fp1, 'fn': fn, 'se': ses}

    
    @property
    def dist_threshold(self):
        return self._dist_threshold

    @property
    def tp_all(self):
        return self._tp

    @property
    def fp1_all(self):
        return self._fp1

    @property
    def fp2_all(self):
        return self._fp2

    @property
    def fp_all(self):
        return self.fp1_all + self.fp2_all

    @property
    def tn_all(self):
        return self._tn

    @property
    def fn_all(self):
        return self._fn

    @property
    def prec(self):
        prec = 0.
        if (self.tp_all + self.fp_all) > 0.:
            prec = self.tp_all / (self.tp_all + self.fp_all)
        return prec

    @property
    def recall(self):
        recall = 0.
        if (self.tp_all + self.fn_all) > 0.:
            recall = self.tp_all / (self.tp_all + self.fn_all)
        return recall

    @property
    def f1(self):
        f1 = 0.
        if self.prec+self.recall > 0.:
            f1 = 2 * self.prec * self.recall / (self.prec + self.recall)
        return f1
    
    @property
    # def accuracy(self):
    #     accuracy = 0.
    #     if self.tp_all+self.tn_all+self.fp_all+self.fn_all > 0.:
    #         accuracy = (self.tp_all+self.tn_all) / (self.tp_all+self.tn_all+self.fp_all+self.fn_all)
    #     return accuracy

    @property
    def sq_errs(self):
        return self._ses

    @property
    def ap(self):
        inds = np.argsort(-1 * np.array(self._scores)).tolist()
        tp   = 0
        r2p  = {}
        for i, ind in enumerate(inds, start=1):
            tp += self._ys[ind]
            p   = tp / i
            r   = tp / (self.tp_all + self.fn_all)
            if not r in r2p.keys():
                r2p[r] = p
            else:
                if r2p[r] < p:
                    r2p[r] = p
        prev_r = 0
        ap = 0.
        for r, p in r2p.items():
            ap += (r-prev_r) * p
            prev_r = r
        return ap

    @property
    def rmse(self):
        _rmse = - np.Inf
        if len(self.sq_errs) > 0:
            _rmse = np.sqrt(np.array(self.sq_errs).mean())
        return _rmse

    def print_results(self, txt=None, elapsed_time=0., num_frames=0, with_ap=True):
        if txt is not None:
            log.info('{}'.format(txt))
        if num_frames > 0:
            log.info('Elapsed time: {}, FPS: {} ({}/{})'.format(elapsed_time, num_frames/elapsed_time, num_frames, elapsed_time))
        if with_ap:
            log.info('| TP   | FP   | FN   | Prec       | Recall       | F1       | AP    |')
            log.info('| ---- | ---- | ---- | ---- | ---------- | ------------ | -------- | ----- |')
            log.info('| {tp} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f}| {ap:.4f} |'.format(tp=self.tp_all,  fp=self.fp1_all, fn=self.fn_all, prec=self.prec, recall=self.recall, f1=self.f1, ap=self.ap))
        else:
            log.info('| TP   | FP   | FN   | Prec       | Recall       | F1       ')
            log.info('| ---- | ---- | ---- | ---------- | ------------ | -------- ')
            log.info('| {tp} | {fp} | {fn} | {prec:.4f} | {recall:.4f} | {f1:.4f} |'.format(tp=self.tp_all,  fp=self.fp1_all, fn=self.fn_all, prec=self.prec, recall=self.recall, f1=self.f1))




