import argparse
from pathlib import Path
from time import time

import numpy as np
from video_reader import PyVideoReader
from decord import VideoReader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking video_reader vs decord")
    parser.add_argument("--filename", "-f", type=str, help="Path to the video file or directory containing video files")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Number of frames in each batch")
    parser.add_argument("--num_batches", "-n", type=int, default=5, help="Number of batches to test")
    return parser.parse_args()


def bench_get_batch(filename: str, indices: np.ndarray):
    start = time()
    vr = PyVideoReader(filename, threads=0)
    vid = vr.get_batch(indices, with_fallback=True)
    duration = time() - start
    return vid, duration


def bench_get_batch_dec(filename: str, indices: np.ndarray):
    start = time()
    vr = VideoReader(filename, num_threads=0)
    vid = vr.get_batch(indices).asnumpy()
    duration = time() - start
    return vid, duration, vr.get_key_indices()


def test_single_file(filename: str, batch_size: int = 32, num_batches: int = 5):
    vr = VideoReader(filename, num_threads=1)
    vr_rs = PyVideoReader(filename)
    num_frames = len(vr)
    batch_sizes = [batch_size] * num_batches
    batch_indices = [np.random.randint(0, num_frames, x) for x in batch_sizes]
    ground_truth = np.array([frame for frame in vr_rs])

    dec_durations = []
    vidrs_durations = []
    for batch in batch_indices:
        vid_dec, dur_dec, key_indices = bench_get_batch_dec(filename, batch)
        vid, dur = bench_get_batch(filename, batch)
        dec_durations.append(dur_dec)
        vidrs_durations.append(dur)

        res = np.array_equal(vid, ground_truth[batch])
        if not res:
            print(f"[ERROR] Failed on batch: {batch}")
            print(f"[ERROR] Key indices: {key_indices}")
            failed_indices = []
            for i in range(len(batch)):
                res = np.array_equal(vid[i], vid_dec[i])
                if not res:
                    failed_indices.append(batch[i])
            print("[ERROR] indices that got incorrect frame:", failed_indices)
            raise RuntimeError("Failed to get the correct frames")
    return vidrs_durations, dec_durations


def compare_results(r):
    vidrs_res = [x[0] for x in r]
    dec_res = [x[1] for x in r]
    vidrs_res_flat = list(np.array(vidrs_res).flat)
    dec_res_flat = list(np.array(dec_res).flat)
    print(f"Average duration for video_reader: {np.mean(vidrs_res_flat):.4f} +-{np.std(vidrs_res_flat):.4f}")
    print(f"Average duration for decord: {np.mean(dec_res_flat):.4f} +-{np.std(dec_res_flat):.4f}")


if __name__ == "__main__":
    args = parse_args()
    filename = Path(args.filename)
    if filename.is_file():
        r = [test_single_file(str(filename), args.batch_size, args.num_batches)]
    elif filename.is_dir():
        vid_list = [str(x) for x in filename.glob("*.avi")] + [str(x) for x in filename.glob("*.mp4")]
        r = []
        for vid in tqdm(vid_list):
            r.append(test_single_file(vid, args.batch_size, args.num_batches))

    compare_results(r)
