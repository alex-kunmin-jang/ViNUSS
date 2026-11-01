"""
KVSS video preprocessing: MP4 + annotation JSON  ->  per-subject HDF5 file.

KVSS: AI Hub, https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=&aihubDataSe=data&dataSetSn=638

Preprocessing follows Appendix B.2 of the paper:
  * NIR video: 640 x 480, 5 fps, stored as MP4.
  * Each frame is converted to grayscale, cropped to a 400 x 400 region
    centred on the bed (rows 20:420, cols 120:520) and resized to 224 x 224
    (bicubic).
  * The stream is segmented into 30-second epochs aligned with the sleep-stage
    annotation, 150 frames per epoch. Epochs with fewer than 150 frames
    (recording boundaries) are last-frame padded; epochs with more are
    truncated to 150.
  * All epochs of one subject are stored in a single HDF5 file.

Expected raw layout (KVSS):
    <input_dir>/<subject>/<subject>_video_01.mp4
    <input_dir>/<subject>/<subject>_annotation.json
        annotation.json["Event"] is a list of 30-second events in temporal
        order; entries whose "Event_Label" is one of
        Wake / N1 / N2 / N3 / REM define the epoch labels.

Output HDF5 keys (read by data/video_dataset.py):
    frames      uint8  (num_epochs, 150, H, W)   chunked per epoch
    labels      int    (num_epochs,)             0=Wake 1=N1 2=N2 3=N3 4=REM
    fps         float
    num_epochs  int
    padded_size int    number of padded frames in the last epoch

Usage:
    python preprocess/video_to_h5.py \
        --input_dir  /path/to/KVSS/raw \
        --output_dir /path/to/kvss_h5/224x224 \
        --size 224 --workers 5
"""
import argparse
import json
import os
import time
from multiprocessing import Process

import cv2
import h5py
import numpy as np

LABEL_DICT = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
EPOCH_SECONDS = 30.0
FRAMES_PER_EPOCH = 150  # 30 s x 5 fps
DEFAULT_EXCLUDE = []


def frame_extraction_and_save(subject, input_dir, output_dir, shape, crop, length=None):
    """Extract frames from one subject's video and save them as an HDF5 file."""
    print(f"[Process {os.getpid()}] Starting subject {subject}")
    t0 = time.time()

    video_path = os.path.join(input_dir, subject, subject + "_video_01.mp4")
    json_path = os.path.join(input_dir, subject, subject + "_annotation.json")
    with open(json_path, "r") as f:
        annotations = json.load(f)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Epoch labels from the annotation file
    labels = [LABEL_DICT[e["Event_Label"]] for e in annotations["Event"] if e["Event_Label"] in LABEL_DICT]
    num_epochs = len(labels)
    print(f"Subject {subject}, total number of epochs {num_epochs}")

    ret, _ = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"{subject}: first frame could not be read")

    if (length is None) or (round(length * fps) > frame_count):
        num_frames = frame_count
    else:
        num_frames = int(length * fps)

    output_path = os.path.join(output_dir, subject + ".h5")
    if os.path.exists(output_path):
        print(f"{subject}: output already exists, skipping")
        cap.release()
        return 0

    y0, y1, x0, x1 = crop
    shape_h, shape_w = shape
    batch_data = []
    batch_count = 0
    frame_cnt = 0
    curr_epoch = 1
    batch_size = FRAMES_PER_EPOCH

    with h5py.File(output_path, "w") as h5file:
        frame_dataset = h5file.create_dataset(
            "frames",
            shape=(num_epochs, FRAMES_PER_EPOCH, shape_h, shape_w),
            dtype=np.uint8,
            chunks=(1, FRAMES_PER_EPOCH, shape_h, shape_w),
        )

        while cap.isOpened():
            ret, curr_frame = cap.read()
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
            if not ret or frame_cnt >= num_frames:
                break

            # Close the current 30-second epoch when its time window is passed
            if curr_time > EPOCH_SECONDS * curr_epoch and curr_epoch - 1 < num_epochs:
                batch_size = len(batch_data)
                if batch_size > FRAMES_PER_EPOCH:
                    batch_data = batch_data[:FRAMES_PER_EPOCH]
                    print(f"{subject}: epoch {curr_epoch} has {batch_size} frames > {FRAMES_PER_EPOCH}, truncated")
                elif batch_size < FRAMES_PER_EPOCH:
                    # last-frame padding
                    batch_data = batch_data + [batch_data[-1]] * (FRAMES_PER_EPOCH - batch_size)
                frame_dataset[curr_epoch - 1] = np.stack(batch_data, axis=0)
                batch_data = []
                curr_epoch += 1
                batch_count += 1

            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            curr_frame = cv2.resize(curr_frame, (shape_w, shape_h), interpolation=cv2.INTER_CUBIC)
            batch_data.append(curr_frame)
            frame_cnt += 1

        # Flush the last (possibly partial) epoch
        if batch_data and curr_epoch - 1 < num_epochs:
            batch_size = len(batch_data)
            if batch_size > FRAMES_PER_EPOCH:
                batch_data = batch_data[:FRAMES_PER_EPOCH]
            elif batch_size < FRAMES_PER_EPOCH:
                batch_data = batch_data + [batch_data[-1]] * (FRAMES_PER_EPOCH - batch_size)
            frame_dataset[curr_epoch - 1] = np.stack(batch_data, axis=0)
            batch_count += 1

        h5file.create_dataset("labels", data=labels)
        h5file.create_dataset("fps", data=fps)
        h5file.create_dataset("num_epochs", data=num_epochs)
        h5file.create_dataset("padded_size", data=FRAMES_PER_EPOCH - batch_size)

    cap.release()
    print(f"[Process {os.getpid()}] {subject} done in {time.time() - t0:.1f}s: "
          f"{batch_count} epochs written, {frame_cnt} frames read, {num_epochs} annotated epochs -> {output_path}")
    return 1


def parse_args():
    p = argparse.ArgumentParser(description="KVSS MP4 + annotation JSON -> per-subject HDF5")
    p.add_argument("--input_dir", required=True, help="Root of raw KVSS data (<input_dir>/<subject>/...)")
    p.add_argument("--output_dir", required=True, help="Where to write <subject>.h5 files")
    p.add_argument("--size", type=int, default=224, help="Output frame size (square). Paper: 224")
    p.add_argument("--crop", type=int, nargs=4, default=[20, 420, 120, 520], metavar=("Y0", "Y1", "X0", "X1"),
                   help="Crop region in the 640x480 frame before resizing. Default: 400x400 bed-centred region")
    p.add_argument("--prefix", default="", help="Only process subject IDs starting with this prefix")
    p.add_argument("--subjects", default=None, help="Optional text file with one subject ID per line (overrides --prefix)")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="Subject IDs to skip")
    p.add_argument("--length", type=float, default=None, help="Optional: only process the first N seconds of each video")
    p.add_argument("--workers", type=int, default=5, help="Number of parallel subject processes")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.subjects:
        with open(args.subjects) as f:
            subjects = [s.strip().replace(".h5", "") for s in f if s.strip()]
    else:
        subjects = sorted(s for s in os.listdir(args.input_dir) if s.startswith(args.prefix))
    print("Total subjects found:", len(subjects))

    existing = set(os.listdir(args.output_dir))
    subjects = [s for s in subjects if s + ".h5" not in existing and s not in set(args.exclude)]
    print("Subjects to process:", len(subjects))

    shape = (args.size, args.size)
    crop = tuple(args.crop)
    active = []
    for subject in subjects:
        if len(active) >= args.workers:
            active[0].join()
            active.pop(0)
        proc = Process(target=frame_extraction_and_save,
                       args=(subject, args.input_dir, args.output_dir, shape, crop, args.length))
        proc.start()
        active.append(proc)
    for proc in active:
        proc.join()


if __name__ == "__main__":
    main()
