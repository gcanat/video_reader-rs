import argparse
from pathlib import Path
from time import perf_counter
from PIL import Image

import numpy as np
from video_reader import PyVideoReader
from torchcodec.decoders import VideoDecoder as TchDecoder
from decord import VideoReader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking video_reader vs decord")
    parser.add_argument("--filename", "-f", type=str, help="Path to the video file or directory containing video files")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Number of frames in each batch")
    parser.add_argument("--num_batches", "-n", type=int, default=5, help="Number of batches to test")
    parser.add_argument("--resize", "-r", type=int, default=None, help="Resize longer side of video")

    return parser.parse_args()


def bench_get_batch(filename: str, indices: np.ndarray, resize: int | None = None):
    start = perf_counter()
    vr = PyVideoReader(filename, threads=0, resize_longer_side=resize)
    vid = vr.get_batch(indices, with_fallback=False)
    duration = perf_counter() - start
    return vid, duration


def bench_get_batch_dec(filename: str, indices: np.ndarray, resize: int | None = None):
    if resize is None:
        resize = -1
    start = perf_counter()
    vr = VideoReader(filename, num_threads=0, width=resize)
    vid = vr.get_batch(indices).asnumpy()
    duration = perf_counter() - start
    return vid, duration, vr.get_key_indices()

def bench_get_batch_tch(filename: str, indices: np.ndarray, resize: int | None = None):
    start = perf_counter()
    vr = TchDecoder(filename, dimension_order="NHWC", num_ffmpeg_threads=8)
    vid = vr.get_frames_at(indices=list(indices))
    duration = perf_counter() - start
    return vid, duration


def test_single_file(filename: str, batch_size: int = 32, num_batches: int = 5, resize: int | None = None):
    vr_rs = PyVideoReader(filename, resize_longer_side=resize)
    num_frames = len(vr_rs)
    batch_sizes = [batch_size] * num_batches
    batch_indices = [np.random.randint(0, num_frames, x) for x in batch_sizes]
    # ground_truth = np.array([frame.numpy() for frame in vr_rs])
    ground_truth = vr_rs.decode().numpy()
    print("ground_truth shape:", ground_truth.shape)

    dec_durations = []
    vidrs_durations = []
    vidtch_durations = []
    for batch in batch_indices:
        vid_dec, dur_dec, key_indices = bench_get_batch_dec(filename, batch, resize=resize)
        print("vid_dec shape:", vid_dec.shape)
        vid, dur = bench_get_batch(filename, batch, resize=resize)
        print("video-rs shape", vid.shape)
        vid_tch, dur_tch = bench_get_batch_tch(filename=filename, indices=batch, resize=resize)
        print("vid_tch shape:", vid_tch.data.shape)
        dec_durations.append(dur_dec)
        vidrs_durations.append(dur)
        vidtch_durations.append(dur_tch)

        res = np.allclose(vid/255., ground_truth[batch]/255.) #, rtol=1e-1, atol=1e-3)
        if not res:
            print(f"[ERROR] Failed on batch: {batch}")
            print(f"[ERROR] Key indices: {key_indices}")
            failed_indices = []
            for i in range(len(batch)):
                res = np.array_equal(vid[i], ground_truth[batch[i]])
                # diff = np.abs(vid[i] - ground_truth[batch[i]])
                img = Image.fromarray(ground_truth[batch[i]])
                img.save(f"frame_diff_{i}.png")
                if not res:
                    failed_indices.append(batch[i])
                    for j in range(len(ground_truth)):
                        if np.allclose(vid[i]/255., ground_truth[j]/255.):
                            print(f"Frame {batch[i]} actually matches frame {j}")
                            break


            print("[ERROR] indices that got incorrect frame:", failed_indices)
            # raise RuntimeError("Failed to get the correct frames")
    return vidrs_durations, dec_durations, vidtch_durations


def compare_results(r):
    vidrs_res = [x[0] for x in r]
    dec_res = [x[1] for x in r]
    tch_res = [x[2] for x in r]
    vidrs_res_flat = list(np.array(vidrs_res).flat)
    dec_res_flat = list(np.array(dec_res).flat)
    tch_res_flat = list(np.array(tch_res).flat)
    print(f"Average duration for video_reader: {np.mean(vidrs_res_flat):.4f} +-{np.std(vidrs_res_flat):.4f}")
    print(f"Average duration for decord: {np.mean(dec_res_flat):.4f} +-{np.std(dec_res_flat):.4f}")
    print(f"Average duration for torchcodec: {np.mean(tch_res_flat):.4f} +-{np.std(tch_res_flat):.4f}")


if __name__ == "__main__":
    args = parse_args()
    filename = Path(args.filename)
    if filename.is_file():
        r = [test_single_file(str(filename), args.batch_size, args.num_batches, args.resize)]
    elif filename.is_dir():
        vid_list = [str(x) for x in filename.glob("*.avi")] + [str(x) for x in filename.glob("*.mp4")]
        r = []
        for vid in tqdm(vid_list):
            r.append(test_single_file(vid, args.batch_size, args.num_batches, args.resize))

    compare_results(r)
