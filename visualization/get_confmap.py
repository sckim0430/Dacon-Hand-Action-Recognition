"""Get Confidence Score Map of PoseC3D Model
"""
import argparse
import os
import os.path as osp
import shutil

import torch
import numpy as np
import cv2

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)


def parse_args():
    """Generate Arguments

    Returns:
        argparse.Namespace : arguments
    """
    parser = argparse.ArgumentParser(
        description='Arguments to Get Confidence Map of PoseC3D Model')

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
        default=0.7,
        help='the threshold of action prediction score')

    parser.add_argument(
        '--pose-config',
        default='../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py',
        help='human pose estimation config file path (from mmpose)')

    parser.add_argument(
        '--pose-checkpoint',
        default=('./work_dirs/hrnet_w48_coco_384x288.pth'),
        help='human pose estimation checkpoint file/url')

    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')

    parser.add_argument(
        '--video_folder',
        default='../../data/Video/',
        help='video folder path')

    parser.add_argument(
        '--video_out_folder',
        default='../../data/Video_Output/',
        help='video output folder path')

    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_name.

    Args:
        video_name (str): The video_name.
    Returns:
        list, list : video frame file list, video path list
    """
    video_name = osp.splitext(osp.basename(video_path))[0]

    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', video_name)

    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')

    vid = cv2.VideoCapture(video_path)

    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0

    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1

        flag, frame = vid.read()

    return frames, frame_paths


def detection_inference(args, frame_paths, det_model):
    """Object Detection Inference

    Args:
        args (dict): arguments
        frame_paths (list): list of frame path
        det_model (class): object detection model class

    Returns:
        list : object detection results
    """

    assert det_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                              'trained on COCO')

    results = []

    print('')
    print('Performing Human Detection for each frame')

    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(det_model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results, pose_model, out_confmap_path):
    """Pose Estimation Inference

    Args:
        args (dict): arguments
        frame_paths (list): list of frame path
        det_results (list): object detection results
        pose_model (class): pose estimation model class
        out_confmap_path (str): confidence score map path
    
    Returns:
        None
    """

    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    #get parameters
    num_frame = len(det_results)
    persons = [len(x) for x in det_results]
    num_person = max(persons) if len(persons) else 0
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)
    h, w, _ = cv2.imread(frame_paths[0]).shape
    sigma = 12

    #Index List to Get Confidence Map in Video
    get_list = [15, 27, 35, 41, 52, 61, 77]

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        if i not in get_list:
            continue

        #Initialization of Confidence Map
        confidence_total_map = np.zeros((h, w), dtype=np.uint8)

        # Align Input Format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(
            pose_model, f, d, format='xyxy')[0]

        #Get Confidence Map of Each Part and Total Body
        for j_idx in range(17):
            confidence_map = np.zeros((h, w), dtype=np.float32)
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']
                box_conf = d[j]['bbox'][-1]
                kp_x, kp_y, kp_conf = kp[j, i, j_idx, :]

                for y in range(h):
                    for x in range(w):
                        #Get Confidence Value with Gaussian Distribution Formula
                        conf_value = np.exp(-((y-kp_y)**2+(x-kp_x)**2) /
                                            (2*sigma**2))*kp_conf*box_conf
                        confidence_map[y, x] = np.max(
                            (confidence_map[y, x], conf_value))

            #Generate Confidence Map
            part_out_confmap_path = os.path.join(out_confmap_path, str(j_idx))

            if not os.path.exists(part_out_confmap_path):
                os.mkdir(part_out_confmap_path)
            part_out_confmap_file = os.path.join(
                part_out_confmap_path, os.path.split(f)[1])

            confidence_map = (confidence_map*255).astype(np.uint8)
            confidence_total_map = np.max(
                (confidence_total_map, confidence_map), axis=0)
            conf_img = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)

            cv2.imwrite(part_out_confmap_file, conf_img)

        #Generate Total Confidence Map
        confmap_img = cv2.applyColorMap(confidence_total_map, cv2.COLORMAP_JET)

        total_out_confmap_path = os.path.join(out_confmap_path, 'total')

        if not os.path.exists(total_out_confmap_path):
            os.mkdir(total_out_confmap_path)

        cv2.imwrite(os.path.join(total_out_confmap_path,
                    os.path.split(f)[1]), confmap_img)

        prog_bar.update()
    return


def main():
    args = parse_args()

    #Initialization Object Detection and Pose Estimation Model
    det_model = init_detector(
        args.det_config, args.det_checkpoint, args.device)
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, args.device)

    for video_name in os.listdir(args.video_folder):
        #Get Confidence Map
        out_confmap_path = os.path.join(
            args.video_out_folder, 'confmap_'+video_name[:-4])

        if not os.path.exists(out_confmap_path):
            os.mkdir(out_confmap_path)

        _, frame_paths = frame_extraction(
            osp.join(args.video_folder, video_name))  # return frame

        det_results = detection_inference(args, frame_paths, det_model)
        pose_inference(args, frame_paths, det_results,
                       pose_model, out_confmap_path)

        torch.cuda.empty_cache()

        tmp_frame_dir = osp.dirname(frame_paths[0])
        shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
