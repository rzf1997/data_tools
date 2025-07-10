import os
import shutil
import decord
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace
from decord import VideoReader, cpu


backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'fastmtcnn',
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

alignment_modes = [True, False]


def split_csv(save_path=None, df=None,n=None):
    parent_dir = Path(save_path).parent
    parent_dir.mkdir(exist_ok=True, parents=True)

    spl_df = np.array_split(df,n)
    for i in range(n):
        sub_df = spl_df[i]
        sub_df.to_csv(save_path.format(i),index=False,header=True)


def merge_csv(df_path=None, n=None, save_path=None):
    skip_list = []
    for i in range(n):
        try:
            sub_df = pd.read_csv(df_path.format(i))
        except:
            skip_list.append(i)
            continue

        if i == 0:
            df = sub_df
        else:
            df = pd.concat([df, sub_df])
    df.to_csv(save_path, index=False, header=True)
    print(f'Skip list: {skip_list}')


def main(args):
    info_path = args.csv_path.format(args.idx)
    info_df = pd.read_csv(info_path)

    print(f'Process {info_path}')

    data_list = []
    for idx, row in tqdm(info_df.iterrows(), total=info_df.shape[0]):
        video_path = row['path']
        caption = row['text']
        face_flag = False

        try:
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()
            for idx in range(len(vr)):
                if idx % int(fps) != 0:
                    continue
                frame = vr[idx].asnumpy()
                detected_face = DeepFace.extract_faces(frame, detector_backend=backends[7], align = alignment_modes[0], enforce_detection=False)
                if detected_face[0]['confidence'] > 0.6:
                    face_flag = True
                    break
        except:
            print(f'Error: {video_path}')
            continue
        
        if face_flag:
            frames = len(vr)
            height = vr[0].shape[0]
            width = vr[0].shape[1]
            data_list.append([video_path, caption, fps, frames, height, width])

            if args.check_dir is not None and len(data_list) < 50:
                save_path = Path(args.check_dir)
                save_path.mkdir(exist_ok=True, parents=True)
                shutil.copy(video_path, str(save_path / Path(video_path).name))
    
    print(f'Number of videos with human face: {len(data_list)}')
    new_df = pd.DataFrame(data_list, columns=['path', 'text', 'fps', 'frames', 'height', 'width'])
    new_df.to_csv(args.output_path.format(args.idx), index=False, header=True)


def main2(args):
    info_path = args.csv_path.format(args.idx)
    info_df = pd.read_csv(info_path)

    print(f'Process {info_path}')

    data_list = []
    for idx, row in tqdm(info_df.iterrows(), total=info_df.shape[0]):
        video_path = row['path']
        caption = row['text']
        append_flag = True

        try:
            vr = VideoReader(video_path)
            fps = vr.get_avg_fps()
            # detect face for the all frames
            for idx in range(len(vr)):
                if idx % int(fps) != 0:
                    continue
                frame = vr[idx].asnumpy()
                detected_face = DeepFace.extract_faces(frame, detector_backend=backends[7], align = alignment_modes[0], enforce_detection=False)
                if len(detected_face) > 1:
                    append_flag = False
                    break
        except:
            print(f'Error: {video_path}')
            continue
        
        if append_flag:
            frames = len(vr)
            height = vr[0].shape[0]
            width = vr[0].shape[1]
            data_list.append([video_path, caption, fps, frames, height, width])

            if args.check_dir is not None and len(data_list) < 50:
                save_path = Path(args.check_dir)
                save_path.mkdir(exist_ok=True, parents=True)
                shutil.copy(video_path, str(save_path / Path(video_path).name))
    
    print(f'Number of videos with human face: {len(data_list)}')
    new_df = pd.DataFrame(data_list, columns=['path', 'text', 'fps', 'frames', 'height', 'width'])
    new_df.to_csv(args.output_path.format(args.idx), index=False, header=True)


def main3(args):
    info_path = args.csv_path.format(args.idx)
    info_df = pd.read_csv(info_path)
    ori_df = pd.read_csv('/project/llmsvgen/share/opensora_datafile/trailer.csv')

    print(f'Process {info_path}')

    data_list = []
    for idx, row in tqdm(info_df.iterrows(), total=info_df.shape[0]):
        path = row['path']
        text = row['text']
        fps = row['fps']
        frames = row['frames']
        height = row['height']
        width = row['width']

        chosen = ori_df[ori_df['video_path'] == path]
        optimal_score = eval(chosen['optimal_score'].values[0])
        avg_optimal_score = chosen['avg_optimal_score'].values[0]
        of_score = chosen['of_score'].values[0]

        data_list.append([path, text, fps, frames, height, width, optimal_score, avg_optimal_score, of_score])
    new_df = pd.DataFrame(data_list, columns=['path', 'text', 'fps', 'frames', 'height', 'width', 'optimal_score', 'avg_optimal_score', 'of_score'])
    new_df.to_csv(args.output_path.format(args.idx), index=False, header=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/project/llmsvgen/share/opensora_datafile/pexels_human_walk_info.csv')
    parser.add_argument('--output_path', type=str, default='/project/llmsvgen/share/opensora_datafile/pexels_human_single_walk_info.csv')
    parser.add_argument('--check_dir', type=str, default=None)
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    # main(args)
    main2(args)
    # main3(args)


    # df_path = '/project/llmsvgen/share/data_process/movies.csv'
    # save_path = '/project/llmsvgen/share/data_process/movies_human/movies_{}.csv'
    # info_df = pd.read_csv(df_path)
    # split_csv(save_path=save_path, df=info_df, n=50)

    # df_path = '/project/llmsvgen/share/data_process/movies_human/movies_human_single_info_{}.csv'
    # save_path = '/project/llmsvgen/share/opensora_datafile/movies_human_single_info.csv'
    # merge_csv(df_path=df_path, n=50, save_path=save_path)