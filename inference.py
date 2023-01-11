"""Action Recognition Single Gpu Inference
"""
from cProfile import label
import time
import argparse
import os
import os.path as osp

import torch
import cv2
import numpy as np

import mmcv
from mmcv import DictAction
from mmaction.apis import inference_recognizer_i, init_recognizer
from mmaction.datasets.pipelines import UniformSampleFrames
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


def parse_args():
    """Generate Arguments

    Returns:
        argparse.Namespace : arguments
    """
    parser = argparse.ArgumentParser(description='Quantom Short Video Demo')

    parser.add_argument(
        '--det-config',
        default='../mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py',
        help='human det config file path (from mmpose)')

    parser.add_argument(
        '--det-checkpoint',
        default=('./work_dirs/faster_rcnn_r50_fpn_1x_coco-person.pth'),
        help='human det checkpoint file/url')

    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='the threshold of action prediction score')

    parser.add_argument(
        '--pose-config',
        default='../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')

    parser.add_argument(
        '--pose-checkpoint',
        default=('./work_dirs/hrnet_w32_coco_256x192.pth'),
        help='human pose estimation checkpoint file/url')

    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py',
        help='skeleton-based action recognition config file path')

    parser.add_argument(
        '--skeleton-checkpoint',
        default='./work_dirs/local/slowonly_r50_u48_240e_ntu120_xsub_keypoint/epoch_1.pth',
        help='skeleton-based action recognition checkpoint file/url')

    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')

    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')

    parser.add_argument(
        '--video_folder',
        default='../../data/Video/',
        help='video folder path')

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


def frame_extraction(video_path):
    """Extract frames given video_name.

    Args:
        video_name (str): The video_name.
    Returns:
        np.ndarray : frame image array
    """

    vid = cv2.VideoCapture(video_path)

    frames = []
    flag, frame = vid.read()
    cnt = 0

    while flag:
        frames.append(frame)
        cnt += 1

        flag, frame = vid.read()

    return np.asarray(frames)


def detection_inference(args, frames, det_model):
    """Object Detection Inference

    Args:
        args (class): arguments
        frames (np.ndarray): frame image array 
        det_model (class): object detection model class

    Returns:
        list: object detection result list
    """

    assert det_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                              'trained on COCO')
    results = []

    print('')
    print('Performing Human Detection for each frame')

    prog_bar = mmcv.ProgressBar(len(frames))
    for frame in frames:
        result = inference_detector(det_model, frame)

        for idx, _ in enumerate(result):
            det = _[0][_[0][:, 4] >= args.det_score_thr]
            results.append(det)

        prog_bar.update()

    return results


def pose_inference(frames, det_results, pose_model):
    """Pose Estimation Inference

    Args:
        frames (np.ndarray): frmae image array
        det_results (list): object detection result list
        pose_model (class): pose estimation model class

    Returns:
        list: pose estimation result list
    """
    print('')
    print('Performing Human Pose Estimation for each frame')

    ret = []
    prog_bar = mmcv.ProgressBar(len(frames))

    for f, d in zip(frames, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(
            pose_model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
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

    #Initialization of Object Detection,Pose Estimation, Action Recognition Model
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, args.device)
    action_recognition_model = init_recognizer(
        config, args.skeleton_checkpoint, args.device)
    det_model = init_detector(
        args.det_config, args.det_checkpoint, args.device)

    for video_name in os.listdir(args.video_folder):
        #Get Frames from Video
        frames = frame_extraction(osp.join(args.video_folder, video_name))
        num_frame = len(frames)
        num_time_step = 48
        h, w, _ = frames[0].shape

        #Apply UniformSampling
        sampler = UniformSampleFrames(
            clip_len=num_time_step, num_clips=1, test_mode=True)
        results = dict(total_frames=num_frame, start_index=0)
        sampling_results = sampler(results)

        torch.cuda.empty_cache()

        #Get Start Time
        start_time = time.time()

        #Get Object Detection and Pose Estimation Results
        det_results = detection_inference(
            args, frames[sampling_results['frame_inds']], det_model)
        pose_results = pose_inference(
            frames[sampling_results['frame_inds']], det_results, pose_model)

        #Generate Action Recognition Input Format
        persons = [len(x) for x in det_results]
        num_person = max(persons) if len(persons) else 0
        num_keypoint = 17

        fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=num_time_step)

        fake_anno['frame_inds'] = np.array(range(num_time_step))
        fake_anno['clip_len'] = sampling_results['clip_len']
        fake_anno['frame_interval'] = sampling_results['frame_interval']
        fake_anno['num_clips'] = sampling_results['num_clips']

        keypoint = np.zeros((num_person, num_time_step, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_time_step, num_keypoint),
                                  dtype=np.float16)

        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]

        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

        #Get Action Recognition Result
        result = inference_recognizer_i(
            action_recognition_model, fake_anno, samples_per_gpu=1)[0][1]

        #Get End time
        end_time = time.time()
        print()
        print('cost time : {}'.format(end_time-start_time))
        print('{} action recognition score : {}'.format(video_name, result))
        print('{} action recognition result : {}'.format(
            video_name, 1 if result >= args.action_score_thr else 0))


if __name__ == '__main__':
    main()
