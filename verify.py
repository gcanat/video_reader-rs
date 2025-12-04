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

# Enable debug logging for the rust library
os.environ['RUST_LOG'] = 'debug'

vid_path = '/Users/bytedance/Downloads/ea09afcd-425a-499e-86c3-a88d0b1d70c8.mov'

batch_indexs = [73, 87, 111, 172, 174, 192, 214, 227, 235, 238, 253, 276, 405, 417, 433, 480, 509, 527, 532, 532, 535, 539, 541, 543, 577, 579, 606, 606, 635, 644, 645, 689]
batch_indexs = [63, 78, 140, 180, 221, 226, 339, 355, 393, 407, 437, 444, 445, 465, 466, 468, 498, 498, 505, 532, 536, 546, 548, 556, 574, 576, 584, 592, 604, 615, 628, 701]

# Full test with mixed indices (from both GOPs - before and after first non-zero keyframe)
# batch_indexs = [28, 45, 63, 78, 140, 180, 221, 226, 339, 355, 393, 407, 437, 444, 445, 465, 466, 468, 498, 498, 505, 532, 536, 546, 548, 556, 574, 576, 584, 592, 604, 615, 628, 701]
# batch_indexs = [61, 78, 151, 155, 179, 189, 213, 262, 301, 327, 328, 330, 378, 387, 405, 432, 451, 454, 473, 494, 591, 615, 627, 658, 665, 709, 730, 732, 735]
batch_indexs = [0, 4, 9, 22, 186, 186, 194, 206, 226, 254, 284, 290, 300, 320, 324, 331, 349, 383, 403, 417, 444, 490, 494, 543, 585, 613, 617, 643, 659, 661, 671, 715]

batch_indexs = [0, 20, 49, 100, 108]

start_time = time.time()
vr_rs = PyVideoReader(vid_path)
# Get PTS info first
pts_list = vr_rs.get_pts()
print(f"Total frames: {len(vr_rs)}")
print(f"First 20 PTS values: {pts_list[:20]}")

# Check PTS order (are they sorted?)
is_sorted = all(pts_list[i] <= pts_list[i+1] for i in range(len(pts_list)-1))
print(f"PTS values are sorted: {is_sorted}")

# Check what the 63rd frame's PTS actually is in ground_truth order (packet order)
print(f"PTS of ground_truth[63] (packet order): {pts_list[63]:.4f} = {pts_list[63] * 600:.0f} timestamp units")
print(f"PTS of ground_truth[56] (packet order): {pts_list[56]:.4f} = {pts_list[56] * 600:.0f} timestamp units")

# Print PTS sequence for frames 55-70
print("\nPTS sequence for frames 55-70 (packet order):")
for i in range(55, 71):
    print(f"  packet {i}: PTS = {pts_list[i]:.4f} = {pts_list[i] * 600:.0f} ts")

# Check if iterator output is in PTS sorted order by examining the actual frames
# This is the key test: is ground_truth[N] the frame with (N+1)th smallest PTS?
sorted_pts = sorted(enumerate(pts_list), key=lambda x: x[1])
print("\nFirst 10 frames in sorted PTS order:")
for rank, (packet_idx, pts_val) in enumerate(sorted_pts[:10]):
    print(f"  rank {rank}: packet {packet_idx}, PTS = {pts_val:.4f}")

# Count frames with negative PTS
negative_pts_count = sum(1 for pts in pts_list if pts < 0)
print(f"\nNumber of frames with negative PTS: {negative_pts_count}")

# Check which packet corresponds to ground_truth[63]
# If iterator outputs in PTS order, ground_truth[63] should match packet at sorted_pts[63]
print(f"\nSorted PTS position 56: packet {sorted_pts[56][0]}, PTS = {sorted_pts[56][1]:.4f}")
print(f"Sorted PTS position 63: packet {sorted_pts[63][0]}, PTS = {sorted_pts[63][1]:.4f}")

# Check PTS around keyframe 56
print(f"PTS around keyframe 56 (packets 50-65):")
for i in range(50, 66):
    print(f"  packet {i}: PTS = {pts_list[i]:.4f}")

# Check video info
info = vr_rs.get_info()
print(f"\nVideo info:")
print(f"  start_time: {info.get('start_time', 'N/A')}")
print(f"  time_base: {info.get('time_base', 'N/A')}")

ground_truth = np.array([frame for frame in vr_rs])
batch_gt = ground_truth[batch_indexs]
end_time = time.time()
print(f"Time taken to get ground truth: {end_time - start_time:.2f} seconds")

start_time = time.time()
vr = PyVideoReader(vid_path)
batch_arr = vr.get_batch(batch_indexs, with_fallback=False)
end_time = time.time()
print(f"Time taken to get batch (with_fallback=False): {end_time - start_time:.2f} seconds")
print(f"with_fallback=False: {np.array_equal(batch_arr, batch_gt)}")

# Compare first frame in detail
print(f"\nFirst index: {batch_indexs[0]}")
print(f"Ground truth shape: {batch_gt[0].shape}, sum: {batch_gt[0].sum()}")
print(f"Batch arr shape: {batch_arr[0].shape}, sum: {batch_arr[0].sum()}")
print(f"First frame match: {np.array_equal(batch_arr[0], batch_gt[0])}")

# Check which frames match
for i, idx in enumerate(batch_indexs):
    if not np.array_equal(batch_arr[i], batch_gt[i]):
        print(f"Frame {idx} mismatch: batch sum={batch_arr[i].sum()}, gt sum={batch_gt[i].sum()}")
        # Try to find which ground truth frame matches (search wider range)
        for j in range(max(0, idx-60), min(len(ground_truth), idx+60)):
            if np.array_equal(batch_arr[i], ground_truth[j]):
                print(f"  -> Actually matches ground_truth[{j}] (offset: {idx - j})")
                break
        else:
            print(f"  -> No match found in range [{max(0,idx-60)}, {min(len(ground_truth), idx+60)})")

start_time = time.time()
vr2 = PyVideoReader(vid_path)
batch_arr2 = vr2.get_batch(batch_indexs, with_fallback=True)
end_time = time.time()
print(f"\nTime taken to get batch (with_fallback=True): {end_time - start_time:.2f} seconds")
print(f"with_fallback=True: {np.array_equal(batch_arr2, batch_gt)}")