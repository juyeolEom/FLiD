import random

import numpy as np
import torch
from torch.utils.data import Dataset

def random_transform(input):
    p_trans = random.randrange(8)
    if p_trans == 0:
        input = input
    elif p_trans == 1:
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 2:
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 3:
        input = np.rot90(input, k=3, axes=(1, 2))
    elif p_trans == 4:
        input = input[:, :, ::-1]
    elif p_trans == 5:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_trans == 6:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_trans == 7:
        input = input[:, :, ::-1]
        input = np.rot90(input, k=3, axes=(1, 2))
    return input


def random_roll(input):
    r_trans = np.random.randint(-2, 2)
    input = np.roll(input, r_trans, axis=0)
    return input


class trainset(Dataset):
    def __init__(self, opt):
        self.path = opt.dataset_path
        self.patch_x = opt.patch_x
        self.multiplication = opt.train_dataset_size

        pulse_path = self.path
        self.pulse_data = np.load(pulse_path)
        if self.pulse_data.ndim != 3 or self.pulse_data.dtype != np.float32:
            raise ValueError('Input must be a float32 array with shape (time, height, width)')
        if opt.patch_x < 16 or opt.patch_x % 16:
            raise ValueError('patch_x must be a positive multiple of 16')
        if min(self.pulse_data.shape[1:]) <= self.patch_x:
            raise ValueError('Training height and width must exceed patch_x')


    def __getitem__(self, index):
        pulse_data = self.pulse_data

        x = np.random.randint(0, pulse_data.shape[1] - self.patch_x, 1)[0]
        y = np.random.randint(0, pulse_data.shape[2] - self.patch_x, 1)[0]
        pulse_data = pulse_data[:, x:x+self.patch_x, y:y+self.patch_x]

        pulse_data = pulse_data-pulse_data.mean()

        pulse_data = random_transform(pulse_data)
        pulse_data = random_roll(pulse_data)

        pulse_data = torch.from_numpy(np.expand_dims(pulse_data, 0).copy())
        return pulse_data

    def __len__(self):
        return self.multiplication


class testset(Dataset):
    def __init__(self, opt, grid=4):
        self.path = opt.dataset_path
        self.patch_x = opt.patch_x // 2
        self.grid = grid
        self.multiplication = (self.grid * 2 - 1) * (self.grid * 2 - 1)

        pulse_path = self.path
        self.pulse_data = np.load(pulse_path)
        if self.pulse_data.ndim != 3 or self.pulse_data.dtype != np.float32:
            raise ValueError('Input must be a float32 array with shape (time, height, width)')
        if opt.patch_x < 16 or opt.patch_x % 16:
            raise ValueError('patch_x must be a positive multiple of 16')

    def __getitem__(self, index):
        pulse_data = self.pulse_data

        x = index // (self.grid * 2 - 1) * self.patch_x // 2
        y = index % (self.grid * 2 - 1) * self.patch_x // 2

        pulse_data = pulse_data[:, x:x+self.patch_x, y:y+self.patch_x]
        mean_ = pulse_data.mean()
        pulse_data = pulse_data - mean_

        pulse_data = torch.from_numpy(np.expand_dims(pulse_data, 0).copy())
        return pulse_data, mean_

    def __len__(self):
        return self.multiplication
