import os
import os.path as osp
import time
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from omegaconf import DictConfig

from utils import save_checkpoint, AverageMeter

log = logging.getLogger(__name__)

def train_epoch(epoch, model, train_loader, loss_criterion, optimizer, device):
    batch_loss = AverageMeter()
    batch_kd_loss = AverageMeter()
    total_loss_meter = AverageMeter()
    model.train()
    # teacher_model.eval()

    t_start = time.time()
    for batch_idx, (imgs, hms) in enumerate(tqdm(train_loader, desc=f'[(TRAIN) Epoch {epoch}]')):
        for scale, hm in hms.items():
            hms[scale] = hm.to(device)

        optimizer.zero_grad()

        preds = model(imgs)

         
        # 原损失函数
        loss = loss_criterion(preds, hms)
        loss.backward()
        optimizer.step()

        batch_loss.update(loss.item(),preds[0].size(0))

    t_elapsed = time.time() - t_start       

    log.info('(TRAIN) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)'.format(epoch=epoch, batch_loss=batch_loss, time=t_elapsed))

    return {'epoch':epoch, 'total_loss':batch_loss.avg}


        
    
    '''
        # 教师模型预测
        with torch.no_grad():
            teacher_preds = teacher_model(imgs)

        # 使用QualityFocalLossWithKD计算总损失
        losses_dict = loss_criterion(preds, hms,teacher_outputs=teacher_preds)

        # 反向传播和优化器更新
        losses_dict['total_loss'].backward()
        optimizer.step()

        # 更新计量器
        batch_loss.update(losses_dict['original_loss'].item(), imgs.size(0))
        batch_kd_loss.update(losses_dict['kd_loss'].item(), imgs.size(0))
        total_loss_meter.update(losses_dict['total_loss'].item(), imgs.size(0))

    t_elapsed = time.time() - t_start

    log.info(f'(TRAIN) Epoch {epoch} Original Loss:{batch_loss.avg:.6f}, '
                f'Knowledge Distillation Loss:{batch_kd_loss.avg:.6f}, '
                f'Total Loss:{total_loss_meter.avg:.6f} Time:{t_elapsed:.1f}(sec)')

    return {
        'epoch': epoch,
        'original_loss': batch_loss.avg,
        'kd_loss': batch_kd_loss.avg,
        'total_loss': total_loss_meter.avg
        }
    
        '''

    



@torch.no_grad()
def test_epoch(epoch, model, dataloader, loss_criterion, device, cfg, vis_dir=None):

    batch_loss    = AverageMeter()
    model.eval()
    
    t_start = time.time()
    for batch_idx, (imgs, hms, trans, xys_gt, visis_gt, img_paths) in enumerate(tqdm(dataloader, desc='[(TEST) Epoch {}]'.format(epoch))):
        imgs = imgs.to(device)
        for scale, hm in hms.items():
            hms[scale] = hm.to(device)
        preds  = model(imgs)
        loss  = loss_criterion(preds, hms)
        batch_loss.update(loss.item(), preds[0].size(0))
    t_elapsed = time.time() - t_start

    log.info('(TEST) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)'.format(epoch=epoch, batch_loss=batch_loss, time=t_elapsed))
    return {'epoch': epoch, 'loss':batch_loss.avg }


