import os
import random

import shutil
from shutil import copytree, ignore_patterns

# 定义原始数据和目标数据的根目录
original_pic_root = '/mnt/data/yuxuan/wasb_data/multi_ball/pic'
original_labels_root = '/mnt/data/yuxuan/wasb_data/multi_ball/label_full'
target_pic_root = '/mnt/data/yuxuan/wasb_data/multiball_full_scene/pic'
target_label_root ='/mnt/data/yuxuan/wasb_data/multiball_full_scene/label'
# 划分比例
split_ratio = 0.8

# 函数：复制文件夹但不包含已有文件
def safe_copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            safe_copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

# 主程序
def split_dataset(original_pic_root, original_labels_root, target_pic_root, target_label_root,split_ratio):
    # 确保目标目录存在
    # train_root = os.path.join(target_root, 'train')
    # val_root = os.path.join(target_root, 'val')
    # os.makedirs(train_root, exist_ok=True)
    # os.makedirs(val_root, exist_ok=True)

    # 遍历原始图片目录中的所有场景文件夹
    for scene in os.listdir(original_pic_root):
        pic_scene_path = os.path.join(original_pic_root, scene)
        labels_scene_path = os.path.join(original_labels_root, scene)

        # 确保场景文件夹存在且非空
        if os.path.isdir(pic_scene_path) and os.path.isdir(labels_scene_path) and not ('copy' in scene):

            # 计算训练集和验证集的文件数量
            files = os.listdir(pic_scene_path)
            num_files = len(files)
            split_point = int(num_files * split_ratio)

            # 分割文件列表
            train_files = files[:split_point]
            val_files = files[split_point:]

            # 复制训练集文件
            train_pic_scene_path = os.path.join(target_pic_root,f'{scene}_train')
            train_labels_scene_path = os.path.join(target_label_root,f'{scene}_train')
            os.makedirs(train_pic_scene_path, exist_ok=True)
            os.makedirs(train_labels_scene_path, exist_ok=True)
            
            for file in train_files:
                src_pic = os.path.join(pic_scene_path, file)
                src_label = os.path.join(labels_scene_path, f'{file}.txt')
                
                dest_pic = os.path.join(train_pic_scene_path, file)
                dest_label = os.path.join(train_labels_scene_path, f'{file}.txt')

                shutil.copytree(src_pic, dest_pic)
                shutil.copy(src_label, dest_label)

            # 复制验证集文件
            val_pic_scene_path = os.path.join(target_pic_root ,f'{scene}_val')
            val_labels_scene_path = os.path.join(target_label_root,f'{scene}_val')
            os.makedirs(val_pic_scene_path, exist_ok=True)
            os.makedirs(val_labels_scene_path, exist_ok=True)
            
            for file in val_files:
                src_pic = os.path.join(pic_scene_path, file)
                src_label = os.path.join(labels_scene_path,f'{file}.txt')
                
                dest_pic = os.path.join(val_pic_scene_path, file)
                dest_label = os.path.join(val_labels_scene_path, f'{file}.txt')

                shutil.copytree(src_pic, dest_pic)
                shutil.copy(src_label, dest_label)


# 调用函数
split_dataset(original_pic_root, original_labels_root, target_pic_root,target_label_root ,split_ratio)
