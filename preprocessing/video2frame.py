"""Convert Video to Frame
"""
import os
import cv2
from tqdm import tqdm

video_root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/Video/'
video2frame_root_dir = '/home/sckim/sckim/3.project/9.Action_Recognition/quantom/data/Video_tmp/'

if not os.path.isdir(video2frame_root_dir):
    os.mkdir(video2frame_root_dir)

for sub_folder_name in tqdm(os.listdir(video_root_dir)):
    sub_folder_path = os.path.join(video_root_dir, sub_folder_name)

    for sub_sub_folder_name in os.listdir(sub_folder_path):
        video_folder_path = os.path.join(sub_folder_path, sub_sub_folder_name)

        for vd in os.listdir(video_folder_path):
            video_path = os.path.join(video_folder_path, vd)
            vd_name, ext = os.path.splitext(vd)

            if ext == '.mp4' or ext == '.avi':
                vid = cv2.VideoCapture(video_path)
                flag, frame = vid.read()

                cnt = 0

                frame_path_format = os.path.join(
                    video2frame_root_dir, sub_folder_name)

                if not os.path.isdir(frame_path_format):
                    os.mkdir(frame_path_format)

                frame_path_format = os.path.join(
                    frame_path_format, sub_sub_folder_name)

                if not os.path.isdir(frame_path_format):
                    os.mkdir(frame_path_format)

                frame_path_format = os.path.join(frame_path_format, vd_name)

                if not os.path.isdir(frame_path_format):
                    os.mkdir(frame_path_format)

                frame_path_format = os.path.join(
                    frame_path_format, 'img_{:06d}.jpg')

                while flag:
                    frame_path = frame_path_format.format(cnt)
                    cv2.imwrite(frame_path, frame)
                    cnt += 1
                    flag, frame = vid.read()

print('done')
