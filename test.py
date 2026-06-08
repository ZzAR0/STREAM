"""Test CoViAR baselines or STREAM and optionally save score npz."""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop, GroupOverSample, GroupScale, StreamGroupCenterCrop, StreamGroupScale


parser = argparse.ArgumentParser(description="STREAM test script")
parser.add_argument("--data-name", type=str, choices=["try"], required=True)
parser.add_argument("--representation", type=str, choices=["iframe", "residual", "mv", "stream"], required=True)
parser.add_argument("--no-accumulation", action="store_true")
parser.add_argument("--data-root", type=str, required=True)
parser.add_argument("--test-list", type=str, required=True)
parser.add_argument("--weights", type=str, required=True)
parser.add_argument("--arch", type=str, default="resnet152")
parser.add_argument("--save-scores", type=str, default=None)
parser.add_argument("--test_segments", type=int, default=8)
parser.add_argument("--test-crops", type=int, default=1, choices=[1, 10])
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("-j", "--workers", default=4, type=int)
parser.add_argument("--gpus", nargs="+", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--save-frames-dir", type=str, default=None)
parser.add_argument("--metrics-summary", type=str, default="./test_accuracy_summary.csv")
parser.add_argument("--warp-direction", type=float, default=None)
args = parser.parse_args()

num_class = 2


def move_to_device(x, device):
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x.to(device, non_blocking=True).contiguous()


def batch_size_of(x):
    if isinstance(x, dict):
        return next(iter(x.values())).size(0)
    return x.size(0)


def aggregate_segments_if_needed(scores, batch_size):
    if scores.size(0) == batch_size:
        return scores
    return scores.view(batch_size, -1, num_class).mean(dim=1)


def load_state(net, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu")
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    if args.representation == "stream" and not any(k.startswith("stream_model.temporal_model.") for k in state.keys()):
        print("Detected legacy STREAM checkpoint without MS-TCN; using identity temporal module for compatibility.")
        net.stream_model.temporal_model = torch.nn.Identity()
    net.load_state_dict(state, strict=True)


def save_accuracy_to_summary(metrics, summary_file):
    headers = [
        "test_list",
        "representation",
        "overall_acc",
        "mAP",
        "acc_class_0",
        "acc_class_1",
        "AP_class_0",
        "AP_class_1",
    ]
    file_path = Path(summary_file)
    if not file_path.exists():
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=headers).writerow({k: metrics.get(k, "") for k in headers})
    print(f"\nMetrics saved to: {summary_file}")


def main():
    warp_direction = args.warp_direction
    if warp_direction is None and os.path.exists(args.weights):
        try:
            checkpoint_meta = torch.load(args.weights, map_location="cpu")
            warp_direction = checkpoint_meta.get("warp_direction")
        except Exception:
            warp_direction = None
    if warp_direction is None:
        warp_direction = -1.0
    net = Model(
        num_class=num_class,
        num_segments=args.test_segments,
        representation=args.representation,
        base_model=args.arch,
        warp_direction=warp_direction,
    )
    load_state(net, args.weights)

    crop_size = net.crop_size if hasattr(net, "crop_size") else args.input_size
    scale_size = net.scale_size if hasattr(net, "scale_size") else int(crop_size * 1.14)

    if args.representation == "stream":
        if args.test_crops != 1:
            raise ValueError("STREAM currently supports --test-crops 1 because mixed-channel oversample needs grouped outputs.")
        cropping = torchvision.transforms.Compose([StreamGroupScale(scale_size), StreamGroupCenterCrop(crop_size)])
    else:
        crop_op = GroupCenterCrop(crop_size) if args.test_crops == 1 else GroupOverSample(
            crop_size, scale_size, is_mv=(args.representation == "mv")
        )
        cropping = torchvision.transforms.Compose([GroupScale(scale_size), crop_op])

    dataset = CoviarDataSet(
        args.data_root,
        args.data_name,
        args.test_list,
        num_segments=args.test_segments,
        representation=args.representation,
        transform=cropping,
        is_train=False,
        accumulate=(not args.no_accumulation),
        save_frames_dir=args.save_frames_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(f"cuda:{args.gpus[0]}" if args.gpus else ("cuda" if torch.cuda.is_available() else "cpu"))
    net = net.to(device).eval()
    if args.gpus and len(args.gpus) > 1:
        net = torch.nn.DataParallel(net, device_ids=args.gpus)

    name_label_list = []
    with open(args.test_list, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    name_label_list.append((parts[0], int(parts[-1])))

    scores_all, labels_all = [], []
    total_infer = 0.0
    for batch_idx, (data, labels) in enumerate(loader):
        data = move_to_device(data, device)
        labels = labels.long()

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                scores = net(data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer += time.time() - start

        bs = batch_size_of(data)
        scores = aggregate_segments_if_needed(scores, bs)
        scores_all.extend(scores.cpu().numpy())
        labels_all.extend(labels.numpy())

        if batch_idx % 10 == 0:
            print(f"Processed {min((batch_idx + 1) * args.batch_size, len(dataset))}/{len(dataset)}")

    scores_arr = np.asarray(scores_all)
    labels_arr = np.asarray(labels_all)
    preds = scores_arr.argmax(axis=1)
    overall_acc = (preds == labels_arr).mean() * 100 if len(labels_arr) else 0.0

    from sklearn.metrics import average_precision_score

    class_acc, ap_list = [], []
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")
    for c in range(num_class):
        idx = labels_arr == c
        acc_c = (preds[idx] == c).mean() * 100 if idx.any() else 0.0
        class_acc.append(acc_c)
        try:
            ap = average_precision_score((labels_arr == c).astype(int), scores_arr[:, c])
        except Exception:
            ap = 0.0
        ap_list.append(ap)
        print(f"Class {c}: acc={acc_c:.2f} AP={ap:.4f}")
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    print(f"mAP: {mAP:.4f}")
    print(f"Average inference time/video: {total_infer / max(len(dataset), 1):.4f}s")

    if args.save_scores:
        names_arr = np.array([x[0] for x in name_label_list[: len(scores_arr)]], dtype=np.str_)
        np.savez(args.save_scores, scores=scores_arr, labels=labels_arr, names=names_arr, overall_accuracy=overall_acc, preds=preds)
        print(f"Saved scores to: {args.save_scores}")

    metrics = {
        "test_list": args.test_list,
        "representation": args.representation,
        "overall_acc": f"{overall_acc:.2f}",
        "mAP": f"{mAP:.4f}",
        "acc_class_0": f"{class_acc[0]:.2f}",
        "acc_class_1": f"{class_acc[1]:.2f}",
        "AP_class_0": f"{ap_list[0]:.4f}",
        "AP_class_1": f"{ap_list[1]:.4f}",
    }
    save_accuracy_to_summary(metrics, args.metrics_summary)
    return metrics


if __name__ == "__main__":
    main()
