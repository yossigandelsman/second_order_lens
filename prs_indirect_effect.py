import numpy as np
import torch
from PIL import Image
import argparse
import json
import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import random

class SwitchLogger(object):
    def __init__(self, model, device, layer, index, mean_ablation):
        self.device = device
        self.layer = layer
        self.index = index
        self.mean_ablation = mean_ablation
        self.model = model
        self.current_layer = 0
        self.switch = True

    def after_gelu(self, ret):
        # bnmhd
        if self.current_layer == self.layer and self.switch:
            ret_new = ret.detach().cpu().numpy()  # [b, n, d]
            ret_new[:, :, self.index] = self.mean_ablation[self.layer, :, self.index]
            self.current_layer += 1
            return torch.from_numpy(ret_new).to(self.device)
        self.current_layer += 1
        return ret

    def reinit(self):
        self.current_layer = 0


def hook_switch_logger(model, device, layer, index, post_gelu, mean_ablation=None):
    """Hooks a projected residual stream logger to the model."""
    prs = SwitchLogger(model, device, layer, index, mean_ablation=mean_ablation)
    if not post_gelu:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.c_fc.post", prs.after_gelu
        )
    else:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.gelu.post", prs.after_gelu
        )
    return prs
