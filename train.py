from pathlib import Path
from types import SimpleNamespace

from ResUNet.engine import run_test, run_train

########################################################################################################################

# Implemented 20261123

# "Waveform-based self-supervised denoising for pulse-sampling fluorescence lifetime imaging"
# https://github.com/juyeolEom/FLiD

########################################################################################################################

opt = SimpleNamespace(
    phase="train",  # "train" or "inference"
    dataset_path="sample001.npy",
    test_model_name="result/2611231823/weight.pth",
    n_epochs=1500,
    seed=2026,
    num_workers=4,
    patch_x=256,
    train_dataset_size=16,
    batch_size=4,
    test_grid=8,
    lr=5e-4,
    b1=0.5,
    b2=0.99,
    lr_decay=0.2,
    lr_decay_patience=100,
    log_interval=50,
)


def main():
    root = Path(__file__).resolve().parent
    opt.dataset_path = root / opt.dataset_path
    opt.test_model_name = root / opt.test_model_name
    opt.result_dir = root / "result"
    opt.test_output_name = opt.test_model_name.parent / (opt.dataset_path.stem + "_denoised.npy")

    if opt.phase == "train":
        run_train(opt)
    elif opt.phase == "inference":
        run_test(opt)
    else:
        raise ValueError('phase must be "train" or "inference"')


if __name__ == "__main__":
    main()
