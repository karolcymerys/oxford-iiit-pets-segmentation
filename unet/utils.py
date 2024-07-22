from typing import Tuple

import torch
from torch.nn.functional import pad


def crop(x: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    input_shape = x.shape
    diff_y = target_shape[2] - input_shape[2]
    diff_x = target_shape[3] - input_shape[3]
    return pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
