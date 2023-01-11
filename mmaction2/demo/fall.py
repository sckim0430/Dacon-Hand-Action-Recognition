# Copyright (c) OpenMMLab. All rights reserved.
from cProfile import label
import time
import argparse
import copy as cp
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.apis import inference_recognizer,init_recognizer
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_detector, build_model, build_recognizer
from mmaction.utils import import_module_error_func

# try:
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                             vis_pose_result)
import warnings
from mmpose.datasets import DatasetInfo

import moviepy.editor as mpy

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/'
        'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='demo/falling.mp4',
        help='video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args

def frame_extraction(video_path,frame_read = 48):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    
    vid = cv2.VideoCapture(video_path)
    frames = [] 
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    fps = 144
    prev_time = 0.0

    while flag:
        current_time = time.time()-prev_time

        if current_time>1./fps:
            prev_time = time.time()
            frames.append(frame)
            frame_path = frame_tmpl.format(cnt + 1)
            frame_paths.append(frame_path)
            cv2.imwrite(frame_path, frame)
                
        
            if cnt%frame_read==0:
                yield frame_paths, frames
                frames.clear()
                frame_paths.clear()

        flag, frame = vid.read()
    
    yield frame_paths,frames


def pose_inference(args,pose_model,frame_paths):
    # model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                         args.device)
    
    ret = []
    # print('Performing Human Pose Estimation for each frame')
    # prog_bar = mmcv.ProgressBar(len(frame_paths))

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)

    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    for f in frame_paths:
        pose = inference_bottom_up_pose_model(pose_model,f,dataset=dataset,dataset_info=dataset_info)[0] 
        ret.append(pose)
        # prog_bar.update()

    return ret

def main():
    args = parse_args()

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.skeleton_config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    model = init_recognizer(config, args.skeleton_checkpoint, args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                args.device)

    vis_frames = []

    start =time.time()
    count_frame = 0
    model_frame_num = 48
    step = model_frame_num

    for index, (frame_paths, original_frames) in enumerate(frame_extraction(args.video,step)):
        num_frame = len(frame_paths)
        count_frame += num_frame
        h, w, _ = original_frames[0].shape
        # Load label_map
        label_map = [x.strip() for x in open(args.label_map,encoding='UTF-8').readlines()]

        pose_results = pose_inference(args,pose_model,frame_paths)
        torch.cuda.empty_cache()

        fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

        num_person = max([len(x) for x in pose_results])
        num_keypoint = 17

        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

        results = inference_recognizer(model, fake_anno)

        action_label = label_map[results[0][0]]
        print('{} action recognition result : {}'.format(index, action_label))

        for i in range(num_frame):
            vis_frames.append(vis_pose_result(pose_model,frame_paths[i],pose_results[i]))
            
            if results[0][1]<0.4:
                continue
            
            cv2.putText(vis_frames[index*model_frame_num+i], action_label, (10, 30), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)        

    end = time.time()
    
    cost_time = end-start
    print('time : ',cost_time)
    print('frame : ', count_frame)
    print('fps : ',count_frame/cost_time)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])

    shutil.rmtree(tmp_frame_dir)

if __name__ == '__main__':
    main()
