from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.spatial.distance import cdist

class KalmanBoxTracker(object):
    count = 0
    
    def __init__(self, center_pos):
        self.center_pos = np.asarray(center_pos).reshape(2,)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)   # State space dimensions: position (x,y) and velocity (vx,vy).
        self.kf.F = np.array([[1., 0., .1, 0.],
                              [0., 1., 0., .1],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.Q *= 0.1
        self.kf.x[:2] = np.array(center_pos)[:, None]
        

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def predict(self):
        prediction = self.kf.predict()
        
        if prediction is None:
            raise ValueError("Prediction returned None. Check the implementation of kf.predict().")
        
        self.center_pos = np.round(prediction[:2]).astype(int)
        
        return prediction[:2]  # 可选：返回预测的中心位置作为反馈

    def update(self, detection_center_pos):
        self.time_since_update = 0
        self.center_pos = detection_center_pos
        self.kf.update(detection_center_pos)

    def get_state(self):
        return tuple(self.center_pos)


class Sort(object):
    def __init__(self, max_age=5, min_hits=3, dist_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold       # Distance threshold for matching.
        self.trackers = []
        self.frame_count = 0

    def update(self, detections_centers):
        self.frame_count += 1
        
        # Predict positions based on existing trackers.
        trks = [t.predict() for t in reversed(self.trackers)]
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections_centers,
                                                                                  list(reversed(trks)),
                                                                                  dist_threshold=self.dist_threshold)

        # Update matched trackers with assigned detections.
        for m in matched:
            self.trackers[len(self.trackers) - m[1]].update(detections_centers[m[0]])

        # Create and initialize new trackers for unmatched detections.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections_centers[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and ( self.frame_count <= self.min_hits):
                yield d
            elif trk.time_since_update > self.max_age:
                self.trackers.pop(i-1)
            i -= 1


def associate_detections_to_trackers(detections_centers, trackers_centers, dist_threshold=50):
    # 当追踪器列表为空时，所有的检测都将被视为未匹配
    if len(trackers_centers) == 0:
        unmatched_detections = list(range(len(detections_centers)))
        return [], unmatched_detections, []

    cost_matrix = pairwise_distances_no_broadcast(detections_centers, trackers_centers)
    
    # 执行匹配前，确认cost_matrix的有效性
    if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
        matches, _ = linear_sum_assignment(cost_matrix)
        matches = [(m[0], len(trackers_centers)-m[1]-1) for m in zip(matches, range(len(matches)))]
    else:
        matches = []

    unmatched_detections = list(set(range(len(detections_centers))) - set([m[0] for m in matches]))
    unmatched_trackers = list(set(range(len(trackers_centers))) - set([m[1] for m in matches]))

    # 过滤掉那些距离超过阈值的匹配
    matches_mask = cost_matrix[[m[0] for m in matches], :][:, [m[1] for m in matches]] < dist_threshold
    
    valid_matches = [match for match, mask in zip(matches, matches_mask.ravel()) if mask]

    return valid_matches, np.array(unmatched_detections), np.array(unmatched_trackers)



def pairwise_distances_no_broadcast(xy_a, xy_b):
    return cdist(xy_a, xy_b)
