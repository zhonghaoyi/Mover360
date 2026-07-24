# Mover360: Controllable Object Manipulation in 360° Panoramic Images

**Project page:** https://zhonghaoyi.github.io/Mover360/

Mover360 performs controllable object manipulation natively on equirectangular (ERP) 360° panoramas — point-, bbox-, and mask-guided object **translation**, **insertion**, and **removal**. It centers on object Translation (relocating a specified object with a single click), with Insert and Remove supported as auxiliary tasks in the same model: a LoRA adaptation of Flux2-Klein-4B-Base conditioned on a three-channel ERP-aligned instruction map and an auxiliary DA² depth map.

Paper and dataset links will be added upon release.

## Environment

Python 3.10. For CUDA 12.6 (stable PyTorch wheels):

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

For CUDA 12.8 / NVIDIA Blackwell GPUs use `requirements_cuda128.txt` instead.

Set `WANDB_MODE=offline` to run without a Weights & Biases account.

## Checkpoints

- `checkpoints/faed.ckpt` — panorama autoencoder used by the FAED evaluation metric (included in this repo).
- Place the trained Mover360 checkpoint anywhere and pass its path to `predict.sh`.

## Data

Training and evaluation expect the UE5 data root at `data/UE5_data` by default; every script accepts a custom path (see `dataset/Mover360_Base.py` for the layout).

## Training

```bash
./train.sh [run_name] [data_dir]
```

Checkpoints and logs are written to `logs/<run_name>/`.

## Prediction

```bash
./predict.sh <move|add|remove> <ckpt> [predict_data_dir] [run_name] [repeat]
```

Results are written to `logs/<run_name>/predict/`.

## Evaluation

```bash
./test.sh <move|add|remove> <result_dir> [test_data_dir] [run_name]
```

Reports FAED, PSNR, SSIM, LPIPS, DINOv3 similarity, and FID.

## Interactive demo

```bash
python interactive.py --help
```

Launches a browser-based editor for point-, bbox-, and mask-guided panorama editing.

<sub>Site template adapted from the [World-Shaper](https://world-shaper-project.github.io/) project page · 360° viewer: [Pannellum](https://pannellum.org/) (MIT)</sub>
