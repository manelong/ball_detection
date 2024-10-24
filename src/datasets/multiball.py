import os
import os.path as osp
import logging
import numpy as np

from utils.dataclasses import Center
log = logging.getLogger(__name__)


def load_csv(csv_path, fids, frame_dir=None):
    with open(csv_path, 'r') as f:
        data = f.read().split('\n')  # ['x1 y1,x2 y2', ...]

    xyvs = {}

    for i in range(len(data)):
        if data[i] == '':  # 'x1 y1,x2 y2'
            break

        centers = []
        for xy in data[i].split(','):  # 'x1 y1'
            es = xy.split(' ')
            x, y = float(es[0]), float(es[1])
            visi = True
            if x == 0 and y == 0:
                visi = False
            centers.append(Center(x=x, y=y, is_visible=visi))

            frame_path = None
            if not frame_dir is None:
                frame_path = osp.join(frame_dir, '{}.jpg'.format(fids[i]))

            xyvs[i] = {'center': centers,
                       'frame_path': frame_path}

    return xyvs


class Multiball(object):
    def __init__(self, cfg):
        self._root_dir = cfg['dataset']['root_dir']
        self._frame_dirname = cfg['dataset']['frame_dirname']
        self._csv_dirname = cfg['dataset']['csv_dirname']
        self._ext = cfg['dataset']['ext']
        self._train_matches = cfg['dataset']['train']['matches']
        self._test_matches = cfg['dataset']['test']['matches']

        self._train_num_clip_ratio = cfg['dataset']['train']['num_clip_ratio']
        self._test_num_clip_ratio = cfg['dataset']['test']['num_clip_ratio']

        self._frames_in = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out']
        self._step = cfg['detector']['step']

        self._load_train = cfg['dataloader']['train']
        self._load_test = cfg['dataloader']['test']

        self._train_all = []
        if self._load_train:
            train_outputs = self._gen_seq_list(self._train_matches, self._train_num_clip_ratio)
            self._train_all = train_outputs['seq_list']
            self._train_num_frames = train_outputs['num_frames']
            self._train_num_frames_with_gt = train_outputs['num_frames_with_gt']
            self._train_num_matches = train_outputs['num_matches']
            self._train_num_rallies = train_outputs['num_rallies']

        self._test_all = []
        if self._load_test:
            test_outputs = self._gen_seq_list(self._test_matches, self._test_num_clip_ratio)
            self._test_all = test_outputs['seq_list']
            self._test_num_frames = test_outputs['num_frames']
            self._test_num_frames_with_gt = test_outputs['num_frames_with_gt']
            self._test_num_matches = test_outputs['num_matches']
            self._test_num_rallies = test_outputs['num_rallies']

        log.info('=> Multiball loaded')
        log.info("Dataset statistics:")
        log.info("-----------------------------------------------------------------------------------")
        log.info("subset       | # batch | # frame | # frame w/ gt | # clip | # game")
        log.info("-----------------------------------------------------------------------------------")
        if self._load_train:
            log.info("train        | {:7d} | {:7d} | {:13d} | {:6d} | {:6d} ".format(
                len(self._train_all), self._train_num_frames, self._train_num_frames_with_gt, self._train_num_rallies,
                self._train_num_matches))

        if self._load_test:
            log.info(
                "test         | {:7d} | {:7d} | {:13d} | {:6d} | {:6d} ".format(
                    len(self._test_all), self._test_num_frames, self._test_num_frames_with_gt, self._test_num_rallies,
                    self._test_num_matches))

        log.info("-----------------------------------------------------------------------------------")

    def _gen_seq_list(self, matches, num_clip_ratio):
        seq_list = []
        clip_seq_list_dict = {}
        clip_seq_gt_dict_dict = {}
        num_frames = 0
        num_matches = len(matches)
        num_rallies = 0
        num_frames_with_gt = 0
        num_clips_no_ball = 0

        for match in matches:
            match_clip_dir = osp.join(self._root_dir, self._frame_dirname, '{}'.format(match))
            clip_names = []

            for clip_name in os.listdir(match_clip_dir):
                if osp.isdir(os.path.join(match_clip_dir, clip_name)):
                    clip_names.append(clip_name)

            clip_names = clip_names[:int(len(clip_names) * num_clip_ratio)]
            num_rallies += len(clip_names)

            for clip_name in clip_names:
                clip_seq_list = []
                clip_seq_gt_dict = {}
                clip_frame_dir = osp.join(self._root_dir, self._frame_dirname, '{}'.format(match), clip_name)
                clip_csv_path = osp.join(self._root_dir, self._csv_dirname, '{}'.format(match), '{}.txt'.format(clip_name))
                fids = []

                for frame_name in os.listdir(clip_frame_dir):
                    if frame_name.endswith(self._ext):
                        fids.append(int(osp.splitext(frame_name)[0]))

                fids.sort()

                ball_xyvs = load_csv(clip_csv_path, fids, frame_dir=clip_frame_dir)  # {0: {'center': centers, 'frame_path': str}, ...

                no_ball = True
                for fid, xyv in ball_xyvs.items():
                    if xyv['center'][0].is_visible:
                        no_ball = False
                if no_ball:  # clips where a ball does not appear are included
                    num_clips_no_ball += 1

                num_frames += len(fids)
                num_frames_with_gt += len(ball_xyvs)

                for i in range(len(ball_xyvs) - self._frames_in + 1):
                    inds = fids[i:i + self._frames_in]
                    names = ['{}{}'.format(ind, self._ext) for ind in inds]
                    paths = [osp.join(clip_frame_dir, name) for name in names]
                    annos = [ball_xyvs[j] for j in range(i, i + self._frames_in)]
                    seq_list.append({'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                    if i % self._step == 0:
                        clip_seq_list.append({'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                clip_seq_list_dict[(match, clip_name)] = clip_seq_list

                for i in range(len(ball_xyvs)):
                    path = osp.join(clip_frame_dir, '{}{}'.format(fids[i], self._ext))
                    clip_seq_gt_dict[path] = ball_xyvs[i]['center']

                clip_seq_gt_dict_dict[(match, clip_name)] = clip_seq_gt_dict

        log.info('{}/{} clips do not include ball trajectory'.format(num_clips_no_ball, len(clip_seq_list_dict)))

        return {'seq_list': seq_list,
                'clip_seq_list_dict': clip_seq_list_dict,
                'clip_seq_gt_dict_dict': clip_seq_gt_dict_dict,
                'num_frames': num_frames,
                'num_frames_with_gt': num_frames_with_gt,
                'num_matches': num_matches,
                'num_rallies': num_rallies}

    @property
    def train(self):
        return self._train_all

    @property
    def test(self):
        return self._test_all
