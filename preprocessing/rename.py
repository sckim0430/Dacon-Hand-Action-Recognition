"""Rename of Video File
"""
import os
from tqdm import tqdm

root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/tmp'
vid_list = os.listdir(root_dir)

#Rename of Video File
for idx, vid_name in tqdm(enumerate(vid_list)):
    old_path = os.path.join(root_dir, vid_name)

    vid_name, ext = os.path.splitext(vid_name)
    new_path = os.path.join(root_dir, 'custom_'+str(idx)+"A002"+ext)
    os.rename(old_path, new_path)
