import argparse
import time
from pathlib import Path
from video_reader import PyVideoReader


try:
    import numpy as np
    from decord import VideoReader
except ImportError:
    raise ImportError("Please install the required packages: `pip install numpy decord tqdm opencv-python`")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking video_reader vs decord")
    parser.add_argument("--filename", "-f", type=str, help="Path to the video file or directory containing video files")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Number of frames in each batch")
    parser.add_argument("--num_batches", "-n", type=int, default=5, help="Number of batches to test")
    parser.add_argument("--num_runs", "-r", type=int, default=3, help="Number of runs for performance testing")
    parser.add_argument(
        "--skip_comparison", "-s", action="store_true", help="Skip frame comparison, only run performance test"
    )
    return parser.parse_args()


def test_decoding_performance(filename: str, num_runs: int = 3):
    """
    Test and compare the decoding performance between PyVideoReader and Decord.
    """
    print("\n===== Performance Testing =====")

    print("Testing PyVideoReader performance...")
    vr_times = []
    vr_frame_count = 0

    for run in range(num_runs):
        start_time = time.time()
        vr_reader = PyVideoReader(filename)
        frame_count = 0
        for frame in vr_reader:
            frame_count += 1
        end_time = time.time()
        elapsed = end_time - start_time
        vr_times.append(elapsed)
        if run == 0:
            vr_frame_count = frame_count
        print(f"  Run {run + 1}: {elapsed:.3f}s, {frame_count} frames, {frame_count / elapsed:.1f} fps")

    vr_avg_time = np.mean(vr_times)
    vr_std_time = np.std(vr_times)
    vr_avg_fps = vr_frame_count / vr_avg_time

    print("\nTesting Decord performance...")
    decord_times = []
    decord_frame_count = 0

    for run in range(num_runs):
        start_time = time.time()
        decord_reader = VideoReader(filename, num_threads=1)
        frame_count = len(decord_reader)
        for i in range(frame_count):
            _ = decord_reader[i].asnumpy()
        end_time = time.time()
        elapsed = end_time - start_time
        decord_times.append(elapsed)
        if run == 0:
            decord_frame_count = frame_count
        print(f"  Run {run + 1}: {elapsed:.3f}s, {frame_count} frames, {frame_count / elapsed:.1f} fps")

    decord_avg_time = np.mean(decord_times)
    decord_std_time = np.std(decord_times)
    decord_avg_fps = decord_frame_count / decord_avg_time

    print("\n===== Performance Summary =====")
    print(f"PyVideoReader:")
    print(f"  Average time: {vr_avg_time:.3f}s ± {vr_std_time:.3f}s")
    print(f"  Average FPS: {vr_avg_fps:.1f}")
    print(f"  Frame count: {vr_frame_count}")

    print(f"\nDecord:")
    print(f"  Average time: {decord_avg_time:.3f}s ± {decord_std_time:.3f}s")
    print(f"  Average FPS: {decord_avg_fps:.1f}")
    print(f"  Frame count: {decord_frame_count}")

    speedup = decord_avg_time / vr_avg_time
    if speedup > 1:
        print(f"\nPyVideoReader is {speedup:.2f}x faster than Decord")
    else:
        print(f"\nDecord is {1 / speedup:.2f}x faster than PyVideoReader")


def compare_videoreader_with_decord(filename: str):
    """
    This function compares the PyVideoReader and Decord libraries by reading frames from a video file
    and checking if the frames match in shape and pixel values. It prints the number of matching frames and
    the differences found, if any.

    Not 100% accurate, but a good sanity check.

    Output using input.mp4:
    Frame count comparison: PyVideoReader=100, Decord=100
    Found 17 differences:
        Diff 1: Frame 83, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.48
        Diff 2: Frame 84, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.52
        Diff 3: Frame 85, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.51
        Diff 4: Frame 86, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.46
        Diff 5: Frame 87, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.31
        Diff 6: Frame 88, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.21
        Diff 7: Frame 89, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 127.03
        Diff 8: Frame 90, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.61
        Diff 9: Frame 91, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.43
        Diff 10: Frame 92, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.37
        Diff 11: Frame 93, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.27
        Diff 12: Frame 94, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.28
        Diff 13: Frame 95, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.23
        Diff 14: Frame 96, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.33
        Diff 15: Frame 97, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.42
        Diff 16: Frame 98, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.57
        Diff 17: Frame 99, Shape match: True, Shapes: (240, 320, 3) vs (240, 320, 3), Pixel diff: 126.69
    """
    print("\n===== Comparing PyVideoReader with Decord =====")
    vr_reader = PyVideoReader(filename)
    decord_reader = VideoReader(filename, num_threads=1)
    vr_frames = []
    decord_frames = []
    print("Reading frames from PyVideoReader...")
    for frame in vr_reader:
        vr_frames.append(frame.copy())
    print("Reading frames from Decord...")
    for i in range(len(decord_reader)):
        decord_frames.append(decord_reader[i].asnumpy())
    print(f"Frame count comparison: PyVideoReader={len(vr_frames)}, Decord={len(decord_frames)}")

    min_frames = min(len(vr_frames), len(decord_frames))
    matches = 0
    differences = []
    print(f"Comparing {min_frames} frames...")

    for i in range(min_frames):
        vr_frame = vr_frames[i]
        decord_frame = decord_frames[i]
        shape_match = vr_frame.shape == decord_frame.shape
        if shape_match:
            diff = np.mean(np.abs(vr_frame.astype(np.float32) - decord_frame.astype(np.float32)))
            content_match = diff < 1.0
        else:
            content_match = False
            diff = float("inf")
        if shape_match and content_match:
            matches += 1
        else:
            differences.append(
                {
                    "frame": i,
                    "shape_match": shape_match,
                    "vr_shape": vr_frame.shape,
                    "decord_shape": decord_frame.shape,
                    "pixel_diff": diff,
                }
            )
    print(f"\nResults: {matches}/{min_frames} frames match exactly ({matches / min_frames * 100:.2f}%)")
    if differences:
        print(f"Found {len(differences)} differences:")
        for i, diff in enumerate(differences):
            print(
                f"  Diff {i + 1}: Frame {diff['frame']}, Shape match: {diff['shape_match']}, "
                f"Shapes: {diff['vr_shape']} vs {diff['decord_shape']}, Pixel diff: {diff['pixel_diff']:.2f}"
            )


if __name__ == "__main__":
    args = parse_args()
    filename = Path(args.filename)

    test_decoding_performance(str(filename), args.num_runs)

    if not args.skip_comparison:
        compare_videoreader_with_decord(str(filename))
