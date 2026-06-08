# STREAM

STREAM is a compressed-domain video classification model built on top of a CoViAR-style MPEG data loader. It loads I-frame, residual, and motion-vector tensors together, aligns I-frame features with motion vectors, and fuses the aligned I-frame and residual features in a single STREAM branch.

This repository intentionally contains only code needed for data loading/extraction, training, testing, and model definition. It does not include datasets, datalists, checkpoints, virtual environments, FFmpeg binaries/source trees, or experiment logs.

## Main Files

- `model.py`: STREAM model, ResNet feature encoders, MV-guided feature warp, temporal fusion, classifier.
- `dataset.py`: CoViAR/STREAM dataset wrapper for compressed videos.
- `data_loader/`: C extension source for loading iframe/residual/MV from compressed videos.
- `train.py`: training entry point.
- `test.py`: evaluation entry point; can save per-video scores.
- `eval_compare_stream.py`: helper for detailed evaluation/comparison.
- `transforms.py`: shared spatial transforms for iframe/residual/MV.

## Environment

Install PyTorch and build the C data loader in your environment. The code expects FFmpeg development headers/libraries to be available to compile `data_loader/coviar_data_loader.c`.

```bash
cd data_loader
bash install.sh
```

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
  --weight-decay 5e-4 \
  --lr-scheduler plateau \
  --patience 3 \
  --factor 0.3 \
  --min-lr 1e-7 \
  --eps 1e-4 \
  --eval-freq 5 \
  --epochs 150 \
  --workers 8 \
  --num_segments 8 \
  --stream-dropout 0.3 \
  --warp-direction 1.0 \
  --gpus 0 \
  --save-dir ./ckpt/experiment_name
```

For the 18+18 STREAM configuration, use `--arch resnet18`; both iframe and residual encoders are ResNet18 in this cleaned tree.

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

## Datalist Format

Each line should point to a video sample and end with its integer class label, matching the original CoViAR-style datalist expected by `dataset.py`.

## Notes

- STREAM mode returns a dict with `iframe`, `residual`, and `mv` tensors.
- In the current 18+18 version, iframe and residual encoders both use the `--arch` backbone.
- `--warp-direction 1.0` was the stable setting in the latest experiments.
