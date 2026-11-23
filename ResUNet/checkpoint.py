from pathlib import Path

import torch


def load_checkpoint(path, net, map_location=None):
    state = torch.load(path, map_location=map_location, weights_only=True)
    net.load_state_dict(state)


def save_checkpoint(net, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model = net.module if isinstance(net, torch.nn.DataParallel) else net
    torch.save(model.state_dict(), path)
    print(f'[Saved] {path}')
