import os
import time
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .dataloader import trainset, testset
from .pulse_denoise_net import PulseDenoiseNet
from .sampling import generate_mask_pair, generate_subimages
from .utils import build_generator, build_mask_generator, get_device, seed_everything, seed_worker


def build_model(opt, device):
    net = PulseDenoiseNet(in_ch=1, filters=[16, 32, 64, 128])
    net.to(device)

    return net


def build_trainloader(opt):
    train_data = trainset(opt)
    worker_init = partial(seed_worker, base_seed=opt.seed)
    return DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        worker_init_fn=worker_init,
        generator=build_generator(opt.seed),
    )


def train_epoch(net, trainloader, optimizer, scheduler, l1, opt, device, mask_generator=None):
    net.train()
    train_loss = 0.0

    for noisy in trainloader:
        f_pulse = noisy.to(device)  # b x c(=1) x pulse_length x patch_x x patch_y
        mask1, mask2, mask3 = generate_mask_pair(f_pulse, generator=mask_generator)
        sub_1 = generate_subimages(f_pulse, mask1)
        sub_2 = generate_subimages(f_pulse, mask2)
        sub_3 = generate_subimages(f_pulse, mask3)

        out = net(sub_1)

        loss2neighbor_1 = 0.5 * l1(sub_2, out) + 0.5 * l1(sub_3, out)

        total_loss = loss2neighbor_1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    scheduler.step(train_loss)

    return train_loss


def run_train(opt):
    seed_everything(opt.seed)

    device = get_device()
    net = build_model(opt, device)
    l1 = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=opt.lr_decay,
        patience=opt.lr_decay_patience,
    )

    trainloader = build_trainloader(opt)

    mask_generator = build_mask_generator(opt.seed, device)
    if opt.n_epochs < 1:
        raise ValueError('n_epochs must be positive')
    run_dir = Path(opt.result_dir) / datetime.now().strftime('%y%m%d%H%M')
    run_dir.mkdir(parents=True, exist_ok=False)

    t0 = time.time()

    for epoch in range(opt.n_epochs):
        train_loss = train_epoch(
            net, trainloader, optimizer, scheduler, l1, opt, device,
            mask_generator=mask_generator
        )

        if epoch % opt.log_interval == 0:
            print('[Epoch %04d/%04d] [Loss: %.4f] [Time Total %.1f]'
                  % (epoch + 1, opt.n_epochs, train_loss, time.time() - t0))

    save_checkpoint(net, run_dir / 'weight.pth')


def test_epoch(net, opt, device):
    output_path = opt.test_output_name
    if not str(output_path).endswith('.npy'):
        output_path = str(output_path) + '.npy'
    if os.path.normcase(os.path.realpath(output_path)) == os.path.normcase(os.path.realpath(opt.dataset_path)):
        raise ValueError('test_output_name must differ from dataset_path; refusing to overwrite the input.')

    checkpoint_path = opt.test_model_name
    load_checkpoint(checkpoint_path, net, map_location=device)

    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    roi_size = opt.patch_x // 2
    grid = opt.test_grid
    test_data = testset(opt, grid)
    input_shape = test_data.pulse_data.shape
    expected_size = roi_size * grid
    if grid < 1 or input_shape[1:] != (expected_size, expected_size):
        raise ValueError(f'Expected spatial shape {(expected_size, expected_size)}; got {input_shape[1:]}')
    pulse_out = np.zeros(input_shape, dtype=np.float32)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for iteration, noisy in enumerate(testloader):
            noisy_input, mean_ = noisy
            noisy_input = noisy_input.to(device)
            noisy_output = net(noisy_input).detach().cpu().numpy()

            x_roi = iteration // (grid * 2 - 1)
            y_roi = iteration % (grid * 2 - 1)
            tmp = noisy_output[0, 0] + mean_.item()

            last_roi = grid * 2 - 2
            if x_roi > 0:
                for k in range(roi_size // 2):
                    tmp[:, k, :] *= (2 * k / roi_size)
            if x_roi < last_roi:
                for k in range(roi_size // 2, roi_size):
                    tmp[:, k, :] *= (2 - 2 * k / roi_size)
            if y_roi > 0:
                for k in range(roi_size // 2):
                    tmp[:, :, k] *= (2 * k / roi_size)
            if y_roi < last_roi:
                for k in range(roi_size // 2, roi_size):
                    tmp[:, :, k] *= (2 - 2 * k / roi_size)

            x0 = x_roi * (roi_size // 2)
            y0 = y_roi * (roi_size // 2)
            pulse_out[:, x0:x0 + roi_size, y0:y0 + roi_size] += tmp

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pulse_out)
    print(f'[Saved] {output_path}')


def run_test(opt):
    seed_everything(opt.seed)
    device = get_device()
    net = build_model(opt, device)
    test_epoch(net, opt, device)
