"""
Compressed-video dataset for CoViAR baselines and STREAM.

For ``representation='stream'`` each sample returns:
    {
        'iframe':   Tensor[T, 3, H, W],
        'residual': Tensor[T, 3, H, W],
        'mv':       Tensor[T, 2, H, W],
    }, label

The three modalities are sampled at matching predictive-frame positions. The
I-frame is loaded from the same GOP as the residual/MV frame so the model can
motion-warp I-frame features toward each predictive frame.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12
REP_TO_INDEX = {"iframe": 0, "mv": 1, "residual": 2}


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ["residual", "mv", "stream"]:
        n -= 1
        n = max(n, 1)

    seg_size = float(n - 1) / num_segments
    if seg_size < 1:
        seg_size = 1.0

    seg_begin = int(np.floor(seg_size * seg))
    seg_end = int(np.floor(seg_size * (seg + 1)))
    if seg_end <= seg_begin:
        seg_end = seg_begin + 1

    seg_begin = max(0, seg_begin)
    seg_end = min(seg_end, n)
    if seg_end <= seg_begin:
        seg_begin = max(0, n - 1)
        seg_end = seg_begin + 1

    if representation in ["residual", "mv", "stream"]:
        seg_begin += 1
        seg_end += 1
        original_n = n + 1
        seg_begin = min(seg_begin, original_n - 1)
        seg_end = min(seg_end, original_n)

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ["residual", "mv", "stream"]:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return max(gop_index, 0), gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(
        self,
        data_root,
        data_name,
        video_list,
        representation,
        transform,
        num_segments,
        is_train,
        accumulate,
        save_frames_dir=None,
    ):
        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate
        self._save_frames_dir = save_frames_dir

        self._input_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        invalid_count = 0
        label_set = set()
        with open(video_list, "r") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    print(f"Warning: bad list line {line_idx}: {line}")
                    invalid_count += 1
                    continue
                video = parts[0]
                try:
                    label = int(parts[-1])
                except ValueError:
                    print(f"Warning: non-integer label at line {line_idx}: {parts[-1]}")
                    invalid_count += 1
                    continue
                if self._data_name == "try" and label not in [0, 1]:
                    print(f"Warning: label out of range at line {line_idx}: {label}")
                    invalid_count += 1
                    continue

                video_path = os.path.join(self._data_root, video[:-4] + ".mp4")
                if not os.path.exists(video_path):
                    print(f"Warning: missing video at line {line_idx}: {video_path}")
                    invalid_count += 1
                    continue
                try:
                    num_frames = get_num_frames(video_path)
                except Exception as exc:
                    print(f"Warning: cannot read frames for {video_path}: {exc}")
                    invalid_count += 1
                    continue
                if num_frames <= 0:
                    print(f"Warning: invalid frame count for {video_path}: {num_frames}")
                    invalid_count += 1
                    continue

                self._video_list.append((video_path, label, num_frames))
                label_set.add(label)

        if not self._video_list:
            raise ValueError("No valid videos loaded. Check data root and list file.")
        print(f"{len(self._video_list)} videos loaded, {invalid_count} invalid skipped.")
        print(f"Labels loaded: {sorted(label_set)}")

    def _get_train_frame_index(self, num_frames, seg, representation=None):
        representation = representation or self._representation
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg, representation)
        rand_end = max(seg_begin, seg_end - 1)
        v_frame_idx = seg_begin if seg_begin >= rand_end else random.randint(seg_begin, rand_end)
        return get_gop_pos(v_frame_idx, representation)

    def _get_test_frame_index(self, num_frames, seg, representation=None):
        representation = representation or self._representation
        original_num_frames = num_frames
        if representation in ["mv", "residual", "stream"]:
            num_frames -= 1
            num_frames = max(num_frames, 1)

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))
        v_frame_idx = max(0, min(v_frame_idx, num_frames - 1))

        if representation in ["mv", "residual", "stream"]:
            v_frame_idx += 1
            v_frame_idx = min(v_frame_idx, original_num_frames - 1)
        return get_gop_pos(v_frame_idx, representation)

    def _load_single(self, video_path, gop_index, gop_pos, representation):
        img = load(video_path, gop_index, gop_pos, REP_TO_INDEX[representation], self._accumulate)
        if img is None:
            print(f"Error: loading {representation} failed for {video_path}, gop={gop_index}, pos={gop_pos}")
            channels = 2 if representation == "mv" else 3
            img = np.zeros((256, 256, channels), dtype=np.uint8)

        if representation == "mv":
            img = clip_and_scale(img, 20)
            img += 128
            img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
        elif representation == "residual":
            img += 128
            img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
        elif representation == "iframe":
            if self._is_train and self._representation == "iframe":
                img = color_aug(img)
            img = img[..., ::-1]
        return img

    def _to_tensor(self, frames, representation):
        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        tensor = torch.from_numpy(frames.copy()).float() / 255.0

        if representation == "iframe":
            tensor = (tensor - self._input_mean) / self._input_std
        elif representation == "residual":
            tensor = (tensor - 0.5) / self._input_std
        elif representation == "mv":
            tensor = tensor - 0.5
        return tensor.contiguous()

    def _getitem_single_representation(self, video_path, num_frames):
        frames = []
        for seg in range(self._num_segments):
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)
            frames.append(self._load_single(video_path, gop_index, gop_pos, self._representation))

        frames = self._transform(frames)
        return self._to_tensor(frames, self._representation)

    def _getitem_stream(self, video_path, num_frames):
        sample = {"iframe": [], "residual": [], "mv": []}
        for seg in range(self._num_segments):
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg, representation="stream")
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg, representation="stream")

            sample["iframe"].append(self._load_single(video_path, gop_index, 0, "iframe"))
            sample["residual"].append(self._load_single(video_path, gop_index, gop_pos, "residual"))
            sample["mv"].append(self._load_single(video_path, gop_index, gop_pos, "mv"))

        sample = self._transform(sample)
        return {key: self._to_tensor(sample[key], key) for key in ["iframe", "residual", "mv"]}

    def __getitem__(self, index):
        if self._is_train:
            video_path, label, num_frames = self._video_list[index]
            if random.random() < 0.5:
                video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        label = int(label)
        if self._data_name == "try":
            label = max(0, min(label, 1))

        if self._representation == "stream":
            return self._getitem_stream(video_path, num_frames), label
        return self._getitem_single_representation(video_path, num_frames), label

    def __len__(self):
        return len(self._video_list)
