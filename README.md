# STREAM

This repository contains the code for the paper "Beyond Pixels: Mining Compressed Domain Artifacts for Efffcient AI-Generated Video Detection" (accepted at ICMl 2026).

## Training Example

```bash
python train.py \
  --lr 0.0003 \
  --batch-size 16 \
  --arch resnet18 \
  --data-name try \
  --representation stream \
  --data-root /path/to/video/root \
  --train-list /path/to/train.txt \
  --test-list /path/to/val.txt \
  --model-prefix STREAM18 \
  --epochs 150 \
  --num_segments 8 \
  --gpus 0 \
  --save-dir ./ckpt/experiment_name
```

## Testing Example

```bash
python test.py \
  --data-name try \
  --representation stream \
  --data-root /path/to/video/root \
  --test-list /path/to/test.txt \
  --weights /path/to/checkpoint.pth \
  --arch resnet18 \
  --test_segments 8 \
  --test-crops 1 \
  --workers 8 \
  --gpus 0 \
  --batch-size 64 \
  --save-scores ./scores.npz \
  --metrics-summary ./metrics.csv
```


