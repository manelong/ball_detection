import os

dir='/mnt/data/yuxuan/wasb_data/multiball_full_scene/label'


for match in os.listdir(dir):
    if 'val' in match:
        print (match+',')