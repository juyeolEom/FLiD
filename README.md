# FLiD

## Setup

Original training environment: Python 3.10.12, PyTorch 2.1.0+cu121,
NumPy 1.26.4, einops 0.8.0, CUDA 12.1.
The release copy has not yet been execution-tested in this environment.

With Python 3.10.12, install the CUDA 12.1 build of PyTorch first:

```sh
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU-only installation, replace `cu121` with `cpu` in the first command.
See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v210).

## Data

Place `sample001.npy` beside `train.py`, or change `dataset_path` to your file name.
Paths are relative to the directory containing `train.py`.
Input must be a preprocessed `float32` NumPy array shaped `(time, height, width)`.
The loader subtracts each patch's mean; it does not perform intensity scaling.
Use the same preprocessing for training and inference.

Training height and width must exceed `patch_x`. Set `patch_x` to a multiple of 16.
Inference expects a square image with side length `patch_x // 2 * test_grid`
(default: 1024). Arbitrary image sizes are not supported.

## Training

Edit the settings at the top of `train.py`, set `phase="train"`, and run:

```sh
python train.py
```

Training starts from scratch and uses one CUDA device if available, otherwise CPU.
The final model weights are saved to `result/YYMMDDHHMM/weight.pth`.
An existing run directory is not overwritten. Optimizer state is not saved.

## Inference

Set `phase="inference"` and `test_model_name` to the relative path of the saved weights,
then run `python train.py`.
The output is saved beside the weights as `<input_name>_denoised.npy`.
Repeated inference on the same input replaces that output.
Weights use a plain model state dictionary; legacy PDN checkpoint dictionaries
are not supported by this loader.

Data and generated results are excluded from Git.
