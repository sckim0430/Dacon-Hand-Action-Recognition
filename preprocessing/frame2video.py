"""Convert Frame to Video
"""

import os
import cv2
from tqdm import tqdm

frame2video_root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/vid2/'
video_root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/'

for folder_name in tqdm(os.listdir(frame2video_root_dir)):
    frame_dir_path = os.path.join(frame2video_root_dir, folder_name)

    #Get Image File Lists
    img_names = os.listdir(frame_dir_path)
    img_names.sort()
    if not len(img_names):
        continue

    #Load Image
    img_path = os.path.join(frame_dir_path, img_names[0])
    img = cv2.imread(img_path)

    h, w, d = img.shape
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    vid_path = os.path.join(video_root_dir, '{}.mp4'.format(folder_name))

    #VideoWriter Initialization
    vd_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    #Video Generate
    for img_name in img_names:
        vd_writer.write(cv2.imread(os.path.join(frame_dir_path, img_name)))

    vd_writer.release()
