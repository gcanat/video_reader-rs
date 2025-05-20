import argparse
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
    return parser.parse_args()


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
    compare_videoreader_with_decord(str(filename))
