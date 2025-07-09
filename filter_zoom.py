import cv2
import decord
from decord import VideoReader, cpu
from pathlib import Path
import numpy as np
import os


def estimate_affine_matrixV2(img1, img2):
    sift = cv2.SIFT_create()
    
    # 检测和描述关键点
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 应用 Lowe's ratio test 来过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # 获取匹配点的坐标,并转换为 2D 点集
    src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2).astype(np.float32)#[:1000]
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2).astype(np.float32)#[:1000]
    # print(src_pts.shape, dst_pts.shape, src_pts.dtype, dst_pts.dtype, np.sum(src_pts), np.sum(dst_pts))


    # M = np.mean(M_list, axis=0)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    return M

def get_diff(vr, interframe=1):
    # vr = VideoReader(video_path, ctx=cpu(0))
    frames = len(vr)
    h_list = []
    count = 0
    for i in range(0, frames, interframe):
        if count == 3:
            break 
        if i + interframe >= frames:
            break
        frame1 = vr[i].asnumpy()
        frame2 = vr[i+interframe].asnumpy()

        # find homography
        if i == 0:
            previous_k = None
        # H, previous_k = findHomography(frame1, frame2, previous_k)
        H = estimate_affine_matrixV2(frame1, frame2)
        # H = estimate_optical_flow(frame1, frame2)
        h_list.append(H)
        count += 1 

    diff_list = []
    for i in range(1, len(h_list)):
        # print(h_list[i], h_list[i-1], type(h_list[i]))
        diff = np.absolute(h_list[i] - h_list[i-1])
        diff_list.append(diff)
    avg_diff = sum(diff_list) / len(diff_list)
    sum_avg_diff = np.sum(avg_diff)

    # sum_avg_diff = sum(h_list) / len(h_list)
    print('sum_avg_diff: ', sum_avg_diff)
    if sum_avg_diff < 0.1:
        return True
    else:
        return False


if __name__ == '__main__':
    v_path = '/home/yxingag/llmsvgen/share/shared_data/panda2m_select/of/of_avg'
    out_root = '/home/yxingag/llmsvgen/share/shared_data/panda2m_select/of/avg_filter'
    for video in os.listdir(v_path):
        print('video: ', video)
        video_path = os.path.join(v_path, video)
        is_zoom = get_diff(video_path)

        if not is_zoom:
            os.system(f'cp {video_path} {out_root}/')
            print(f'cp {video_path} {out_root}/')

    print('Finish')
