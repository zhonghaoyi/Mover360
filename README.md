# Mover360: Controllable Object Manipulation in 360° Panoramic Images

**Project page:** https://zhonghaoyi.github.io/Mover360/

Mover360 performs controllable object manipulation natively on equirectangular (ERP) 360° panoramas — point-, bbox-, and mask-guided object **translation**, **insertion**, and **removal**. It centers on object Translation (relocating a specified object with a single click), with Insert and Remove supported as auxiliary tasks in the same model: a LoRA adaptation of Flux2-Klein-4B-Base conditioned on a three-channel ERP-aligned instruction map and an auxiliary DA² depth map.

**Test benchmark:** [HaoyiZhong/Mover360-benchmark](https://huggingface.co/datasets/HaoyiZhong/Mover360-benchmark) · Paper link will be added upon release.

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
- The trained Mover360 checkpoint will be released separately; place it anywhere and pass its path via `--ckpt` (used by `predict.sh` and `interactive.py`).
- The Flux2-Klein-4B base model and the DA² depth model are downloaded automatically from the Hugging Face Hub on first run; run `huggingface-cli login` first if the base model requires accepting its license.

## Data

Training and evaluation expect the UE5 data root at `data/UE5_data` by default; every script accepts a custom path (see `dataset/Mover360_Base.py` for the layout).

The dual test benchmark is available at [HaoyiZhong/Mover360-benchmark](https://huggingface.co/datasets/HaoyiZhong/Mover360-benchmark): `Test_all_new.zip` (210 held-out UE5 synthetic editing tuples) and `Test_real_all.zip` (50 real captured editing tuples). Unzip them under `data/UE5_data/`, e.g.:

```bash
hf download HaoyiZhong/Mover360-benchmark --repo-type dataset --local-dir data/benchmark
unzip data/benchmark/Test_all_new.zip -d data/UE5_data/
unzip data/benchmark/Test_real_all.zip -d data/UE5_data/
```

### Depth conditioning maps

Mover360 conditions on DA² depth maps stored as a `MovieRenders_PredictDepth`
directory next to `MovieRenders_Normal` inside each data folder. If a data
folder does not contain them yet, precompute them once (the DA² weights are
downloaded automatically from the Hugging Face Hub):

```bash
python depth/DA-2/infer_npy_batch.py --input_root data/UE5_data/Test_all_new/Saved
python depth/DA-2/infer_npy_batch.py --input_root data/UE5_data/Test_real_all/Saved
```

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
python interactive.py --ckpt /path/to/mover360.ckpt --port 7860
```

Launches a browser-based editor for point-, bbox-, and mask-guided panorama editing (results are written to `logs/interactive/`). Requires a CUDA GPU (roughly 24 GB of memory; on GPUs without bf16 support the demo falls back to fp16 automatically). Set `MOVER360_HF_CACHE` or pass `--huggingface-cache` to use a non-default Hugging Face cache location. See `python interactive.py --help` for all options.

<sub>Site template adapted from the [World-Shaper](https://world-shaper-project.github.io/) project page · 360° viewer: [Pannellum](https://pannellum.org/) (MIT)</sub>
