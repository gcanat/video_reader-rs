import os
from time import time
import numpy as np
from PIL import Image
from video_reader import PyVideoReader
import math
from tqdm import tqdm
import contextlib
import sys
import time
import logging
from tqdm import tqdm

@contextlib.contextmanager
def suppress_stderr():
    # supresss warnings by FFmpeg like `[h264 @ 0x11b00cc0] mmco: unref short failure`
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w") as devnull:
        old_stderr_fd = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr_fd, stderr_fd)
            os.close(old_stderr_fd)

# Enable debug logging for the rust library (SLOW! comment out for benchmarking)
# os.environ['RUST_LOG'] = 'debug'

vid_path = '/Users/bytedance/Downloads/ea09afcd-425a-499e-86c3-a88d0b1d70c8.mov'  # badcase, 负 pts, 负 dts
# vid_path = '/Users/bytedance/Downloads/a.mp4'
# vid_path = '/Users/bytedance/Downloads/v10033g50000d4g0nmfog65po93qnbj0 (1) 2.mp4'  # av1 编码
# vid_path = '/Users/bytedance/Downloads/8.mp4'
# vid_path = '/Users/bytedance/Downloads/vid0.mp4'
vid_path = '/Users/bytedance/Downloads/0a7ef2bd-4852-45c6-be96-645972ab2905.mp4'  # badcase

# vid_path = '/Users/bytedance/Downloads/out_A_negpts.mp4'  # 开头负 PTS
# vid_path = '/Users/bytedance/Downloads/out_B_negpts.mp4'  # 拼接点负 PTS
# vid_path = '/Users/bytedance/Downloads/out_C_negpts.mp4'  # 开头+中间双负 PTS 且各不同

# 新 hard case
# vid_path = '/Users/bytedance/Downloads/1_negpts_A.mp4'
# vid_path = '/Users/bytedance/Downloads/2_negpts_B.mp4'
# vid_path = '/Users/bytedance/Downloads/3_negpts_C.mp4'
# vid_path = '/Users/bytedance/Downloads/4_negdts_D.mp4'
# vid_path = '/Users/bytedance/Downloads/5_negdts_E.mp4'
# vid_path = '/Users/bytedance/Downloads/6_negdts_F.mp4'
# vid_path = '/Users/bytedance/Downloads/7_negpts_negdts_G.mp4'
# vid_path = '/Users/bytedance/Downloads/8_negpts_negdts_H.mp4'
# vid_path = '/Users/bytedance/Downloads/9_negpts_negdts_I.mp4'



start_time = time.time()

with suppress_stderr():
# if 1:
    vr_rs = PyVideoReader(vid_path)
    
    # parse nframes
    vid_info = vr_rs.get_info()
    fps = float(vid_info["fps"])
    duration = float(vid_info["duration"])
    calibrated_nframes = int(round(fps * duration))
    num_frames = min(calibrated_nframes, len(vr_rs))
    
    s = time.time()
    actual_frames = vr_rs.count_actual_frames()
    e = time.time()
    print(f"Time taken to count actual frames: {e - s:.2f} seconds")
    
    print(f"num_frames: {num_frames} | {calibrated_nframes=} vs {len(vr_rs)=} vs {actual_frames=}")
    
    # np.random.seed(42)  # Fixed seed for reproducibility
    # batch_indexs = np.sort(np.random.randint(0, num_frames, 500))
    batch_indexs = np.sort(np.random.choice(np.arange(0, num_frames), size=50, replace=False))
    # batch_indexs = np.arange(0, num_frames, 2)
    # batch_indexs = np.array([600])

    ground_truth = np.array([frame for frame in vr_rs])
    batch_gt = ground_truth[batch_indexs]
    end_time = time.time()
    print(f"Time taken to get ground truth: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    vr2 = PyVideoReader(vid_path)
    start_time2 = time.time()
    batch_arr2 = vr2.get_batch(batch_indexs, with_fallback=True)
    end_time = time.time()
    print(f"\nTime taken to get batch (with_fallback=True): {end_time - start_time:.2f} seconds (exclude init time: {end_time - start_time2:.2f} seconds)")
    print(f"with_fallback=True: {np.array_equal(batch_arr2, batch_gt)}")
    
    assert np.array_equal(batch_arr2, batch_gt), f"batch_indexs: {batch_indexs}"

    start_time = time.time()
    vr = PyVideoReader(vid_path)
    start_time2 = time.time()
    batch_arr = vr.get_batch(batch_indexs, with_fallback=False)
    end_time = time.time()
    print(f"Time taken to get batch (with_fallback=False): {end_time - start_time:.2f} seconds (exclude init time: {end_time - start_time2:.2f} seconds)")
    print(f"with_fallback=False: {np.array_equal(batch_arr, batch_gt)}")
    
    assert np.array_equal(batch_arr, batch_gt), f"batch_indexs: {batch_indexs}"
    
    start_time = time.time()
    vr3 = PyVideoReader(vid_path)
    start_time2 = time.time()
    batch_arr3 = vr3.get_batch(batch_indexs, with_fallback=None)
    end_time = time.time()
    print(f"\nTime taken to get batch (with_fallback=None): {end_time - start_time:.2f} seconds (exclude init time: {end_time - start_time2:.2f} seconds)")
    print(f"with_fallback=None: {np.array_equal(batch_arr3, batch_gt)}")
    
    assert np.array_equal(batch_arr3, batch_gt), f"batch_indexs: {batch_indexs}"