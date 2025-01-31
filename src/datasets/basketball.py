import os
import os.path as osp
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from utils.dataclasses import Center
log = logging.getLogger(__name__)

def load_xml(xml_path, frame_names=None, frame_dir=None):
    if frame_names is None:
        assert 0, 'frames_names is None'

    tree = ET.parse(xml_path)
    root = tree.getroot()
    xyvs = {}
    for child in root:
        if child.tag!='track':
            continue

        for child2 in child:
            if child2.tag!='points':
                continue
            fid = int( child2.attrib['frame'] )
            is_outside = True if child2.attrib['outside']=='1' else False
            visi       = True if child2.attrib['occluded']=='0' else False

            if len( child2.attrib['points'].split(';') ) > 1:
                log.info('multiple points exist: {} @ {}'.format(child2.attrib['points'], xml_path))
            pts = child2.attrib['points'].split(';')[0].split(',')
            x, y = float(pts[0]), float(pts[1])

            frame_path = osp.join(frame_dir, '{:06d}.jpg'.format(fid))
            xyvs[fid]  = {'frame_path': frame_path, 
                          'center': Center(is_visible=visi, x=x, y=y),
                          }

    xyvs2 = {}
    for frame_name in frame_names:
        ind = int( osp.splitext(frame_name)[0] )
        if ind in xyvs.keys():
            xyvs2[ind] = xyvs[ind]
        else:
            frame_path = osp.join(frame_dir, frame_name)
            xyvs2[ind] = {'frame_path': frame_path, 
                          'center': Center(is_visible=False, x=-1., y=-1),
                          }

    return xyvs2

def _xml_path_from_dir(directory):
    fnames = os.listdir(directory)
    if len(fnames)!=1:
        log.info('# of files are not 1 @ {}. {}'.format(directory, fnames))
        return None
    if osp.splitext(fnames[0])[-1]!='.xml':
        log.info('.xml does not exist')
        return None
    return osp.join(directory, fnames[0])

def _get_videos_as_diff(src_dir, vnames):
    vnames_all = os.listdir(src_dir)
    vnames_all.sort()
    vnames_out = list( set(vnames_all) - set(vnames) )
    vnames_out.sort()
    return vnames_out

def get_clips(cfg, train_or_test='test', gt=True):
    root_dir      = cfg['dataset']['root_dir']
    video_dirname = cfg['dataset']['video_dirname']
    anno_dirname  = cfg['dataset']['anno_dirname']
    videos        = cfg['dataset'][train_or_test]['videos']
    if (train_or_test=='train') and (videos is None):
        videos = _get_videos_as_diff(osp.join(root_dir, video_dirname), cfg['dataset']['test']['videos'])

    clip_dict = {}
    for video in videos:
        video_dir  = osp.join(root_dir, video_dirname, '{}'.format(video))
        anno_dir   = osp.join(root_dir, anno_dirname, '{}'.format(video))
        if not osp.exists(anno_dir):
            log.info('[ALERT] {} not exists'.format(anno_dir))
            continue
        clip_names = os.listdir(video_dir)
        xml_names = os.listdir(anno_dir)
        xml_name_dict = {}
        for xml_name in xml_names:
            tmp = int(osp.splitext(xml_name)[0].split('_')[-1])
            xml_name_dict[ osp.splitext(xml_name)[0].split('_')[-1] ] = xml_name
 
        for clip_name in clip_names:
            clip_frame_dir = osp.join(video_dir, clip_name)
            if not osp.isdir(clip_frame_dir):
                continue
            clip_xml_path = osp.join(anno_dir, xml_name_dict[clip_name])
            frame_names = os.listdir(clip_frame_dir)
            frame_names.sort()
            ball_xyvs = load_xml(clip_xml_path, frame_dir=clip_frame_dir, frame_names=frame_names)
            clip_dict[(video, clip_name)] = {'clip_dir_or_path': clip_frame_dir, 'clip_gt_dict': ball_xyvs, 'frame_names': frame_names}
    return clip_dict

class Basketball(object):
    def __init__(self, cfg):
        self._root_dir      = cfg['dataset']['root_dir']
        self._video_dirname = cfg['dataset']['video_dirname']
        self._anno_dirname  = cfg['dataset']['anno_dirname']
        self._train_videos = cfg['dataset']['train']['videos']
        self._test_videos  = cfg['dataset']['test']['videos']
        if self._train_videos is None:
            self._train_videos = _get_videos_as_diff(osp.join(self._root_dir, self._video_dirname), cfg['dataset']['test']['videos'])

        self._train_num_clip_ratio = cfg['dataset']['train']['num_clip_ratio']
        self._test_num_clip_ratio  = cfg['dataset']['test']['num_clip_ratio']

        self._frames_in  = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out']
        self._step       = cfg['detector']['step']

        self._load_train      = cfg['dataloader']['train']
        self._load_test       = cfg['dataloader']['test']
        self._load_train_clip = cfg['dataloader']['train_clip']
        self._load_test_clip  = cfg['dataloader']['test_clip']

        self._train_all = []
        self._train_clips       = {}
        self._train_clip_gts    = {}
        self._train_clip_disps  = {}
        if self._load_train or self._load_train_clip:
            train_outputs = self._gen_seq_list(self._train_videos, self._train_num_clip_ratio)
            self._train_all                = train_outputs['seq_list']
            self._train_num_frames         = train_outputs['num_frames']
            self._train_num_frames_with_gt = train_outputs['num_frames_with_gt']
            self._train_num_matches        = train_outputs['num_games']
            self._train_num_rallies        = train_outputs['num_clips']
            self._train_disp_mean          = train_outputs['disp_mean']
            self._train_disp_std           = train_outputs['disp_std']
            if self._load_train_clip:
                self._train_clips       = train_outputs['clip_seq_list_dict']
                self._train_clip_gts    = train_outputs['clip_seq_gt_dict_dict']
                self._train_clip_disps  = train_outputs['clip_seq_disps']

        self._test_all = []
        self._test_clips       = {}
        self._test_clip_gts    = {}
        self._test_clip_disps  = {}
        if self._load_test or self._load_test_clip:
            test_outputs  = self._gen_seq_list(self._test_videos, self._test_num_clip_ratio)
            self._test_all                 = test_outputs['seq_list']
            self._test_num_frames          = test_outputs['num_frames']
            self._test_num_frames_with_gt  = test_outputs['num_frames_with_gt']
            self._test_num_matches         = test_outputs['num_games']
            self._test_num_rallies         = test_outputs['num_clips']
            self._test_disp_mean           = test_outputs['disp_mean']
            self._test_disp_std            = test_outputs['disp_std']
            if self._load_test_clip:
                self._test_clips               = test_outputs['clip_seq_list_dict']
                self._test_clip_gts            = test_outputs['clip_seq_gt_dict_dict']
                self._test_clip_disps          = test_outputs['clip_seq_disps']

        # display stats
        log.info('=> Basketball loaded' )
        log.info("Dataset statistics:")
        log.info("-------------------------------------------------------------------------------------")
        log.info("subset           | # batch | # frame | # frame w/ gt | # clip | # game | disp.[pixel]")
        log.info("-------------------------------------------------------------------------------------")
        if self._load_train:
            log.info("train            | {:7d} | {:7d} | {:13d} | {:6d} | {:6d} | {:2.1f}+/-{:2.1f} ".format(len(self._train_all), self._train_num_frames, self._train_num_frames_with_gt, self._train_num_rallies, self._train_num_matches, self._train_disp_mean, self._train_disp_std ) )
        if self._load_train_clip:
            num_batches_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, clip in self._train_clips.items():
                num_batches  = len(clip)
                num_frames = 0
                for tmp in clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                #print(key)
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._train_clip_disps[key] )
                #disps = np.array([1,2])
                log.info("{} | {:7d} | {:7d} | {:13d} |        |        | {:2.1f}+/-{:2.1f}".format(clip_name, num_batches, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
                num_batches_all          += num_batches
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all          | {:7d} | {:7d} | {:13d} | {:6d} |        | {:2.1f}+/-{:2.1f}".format(num_batches_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        if self._load_test:
            log.info("test             | {:7d} | {:7d} | {:13d} | {:6d} | {:6d} | {:2.1f}+/-{:2.1f} ".format(len(self._test_all), self._test_num_frames, self._test_num_frames_with_gt, self._test_num_rallies, self._test_num_matches, self._test_disp_mean, self._test_disp_std) )
        if self._load_test_clip:
            num_batches_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, clip in self._test_clips.items():
                num_batches  = len(clip)
                num_frames = 0
                for tmp in clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._test_clip_disps[key] )
                log.info("{} | {:7d} | {:7d} | {:13d} |        |        | {:2.1f}+/-{:2.1f}".format(clip_name, num_batches, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
                num_batches_all          += num_batches
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all          | {:7d} | {:7d} | {:13d} | {:6d} |        | {:2.1f}+/-{:2.1f}".format(num_batches_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        log.info("-------------------------------------------------------------------------------------")

    def _gen_seq_list(self, video_names, num_clip_ratio):
        seq_list           = []
        clip_seq_list_dict = {}
        clip_seq_gt_dict_dict = {}
        clip_seq_disps        = {}
        num_frames         = 0
        num_games        = 0
        num_clips        = 0
        num_frames_with_gt = 0
        disps              = []

        for video_name in video_names:

            video_dir = osp.join(self._root_dir, self._video_dirname, '{}'.format(video_name))
            anno_dir  = osp.join(self._root_dir, self._anno_dirname, '{}'.format(video_name))
            if not osp.exists(anno_dir):
                log.info('[ALERT] {} not exists'.format(anno_dir))
                continue
            num_games += 1

            clip_names = os.listdir(video_dir)
            clip_names.sort()
            clip_names = clip_names[:int(len(clip_names)*num_clip_ratio)]

            xml_names = os.listdir(anno_dir)
            xml_name_dict = {}
            for xml_name in xml_names:
                tmp = int(osp.splitext(xml_name)[0].split('_')[-1])
                xml_name_dict[ osp.splitext(xml_name)[0].split('_')[-1] ] = xml_name
            
            for clip_name in clip_names:               
                clip_frame_dir = osp.join(video_dir, clip_name)
                if not osp.isdir(clip_frame_dir):
                    continue
                if not clip_name in xml_name_dict.keys():
                    raise IOError('{} does not exist in {}'.format(clip_name, video_dir))

                num_clips += 1
                clip_seq_list    = []
                clip_seq_gt_dict = {}

                clip_xml_path = osp.join(anno_dir, xml_name_dict[clip_name])

                frame_names = os.listdir(clip_frame_dir)
                frame_names.sort()
                ball_xyvs = load_xml(clip_xml_path, frame_dir=clip_frame_dir, frame_names=frame_names)
                
                is_included = False
                for fid, xyv in ball_xyvs.items():
                    if xyv['center'].is_visible:
                        is_included = True
                if not is_included:
                    log.info('[ALERT] {}/{} excluded since ball does not appeared.'.format(video_name, clip_name))
                    continue

                fids                = list(ball_xyvs.keys())
                num_frames         += len(frame_names)
                num_frames_with_gt += len(ball_xyvs)
            
                for i in range(len(ball_xyvs)-self._frames_in+1):
                    inds  = fids[i:i+self._frames_in]
                    names = [frame_names[j] for j in inds]
                    paths = [ osp.join(clip_frame_dir, name) for name in names]
                    annos = [ ball_xyvs[j] for j in range(i,i+self._frames_in)]
                    seq_list.append( {'frames': paths, 'annos': annos, 'match': 0, 'clip': video_name})
                    if i%self._step==0:
                        clip_seq_list.append( {'frames': paths, 'annos': annos, 'match': 0, 'clip': video_name})
                clip_seq_list_dict[(video_name, clip_name)] = clip_seq_list

                clip_disps = []
                # compute diplacement between consecutive frames
                for i in range(len(ball_xyvs)-1):
                    xy1, visi1 = ball_xyvs[i]['center'].xy, ball_xyvs[i]['center'].is_visible
                    xy2, visi2 = ball_xyvs[i+1]['center'].xy, ball_xyvs[i+1]['center'].is_visible
                    if visi1 and visi2:
                        disp = np.linalg.norm(np.array(xy1)-np.array(xy2))
                        disps.append(disp)
                        clip_disps.append(disp)

                for i in range(len(ball_xyvs)):
                    path     = ball_xyvs[i]['frame_path']
                    clip_seq_gt_dict[path] = ball_xyvs[i]['center']
                clip_seq_gt_dict_dict[(video_name, clip_name)] = clip_seq_gt_dict
                clip_seq_disps[(video_name, clip_name)]         = clip_disps

        return { 'seq_list': seq_list, 
                 'clip_seq_list_dict': clip_seq_list_dict,
                 'clip_seq_gt_dict_dict': clip_seq_gt_dict_dict,
                 'clip_seq_disps': clip_seq_disps,
                 'num_frames': num_frames, 
                 'num_frames_with_gt': num_frames_with_gt, 
                 'num_games': num_games, 
                 'num_clips': num_clips,
                 'disp_mean': np.mean(np.array(disps)),
                 'disp_std': np.std(np.array(disps))}

    @property
    def train(self):
        return self._train_all

    @property
    def test(self):
        return self._test_all

    @property
    def train_clips(self):
        return self._train_clips
    
    @property
    def train_clip_gts(self):
        return self._train_clip_gts

    @property
    def test_clips(self):
        return self._test_clips

    @property
    def test_clip_gts(self):
        return self._test_clip_gts

