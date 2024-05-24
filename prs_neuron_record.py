import numpy as np
import torch


class SwitchLogger(object):
    def __init__(self, model, device, layers, indices):
        self.device = device
        self.layers = layers
        self.indices = indices
        self.model = model
        self.current_layer = 0
        self.recorded = []

    def after_gelu(self, ret):
        # bnmhd
        for layer, index in zip(self.layers, self.indices):
            if self.current_layer == layer:
                ret_new = ret.detach().cpu().numpy()  # [b, n, d]
                self.recorded.append(ret_new[:, :, index])
        self.current_layer += 1
        return ret

    def reinit(self):
        self.current_layer = 0
        self.recorded = []


def hook_switch_logger(model, device, layers, indices, post_gelu=True):
    """Hooks a projected residual stream logger to the model."""
    prs = SwitchLogger(model, device, layers, indices)
    if not post_gelu:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.c_fc.post", prs.after_gelu
        )
    else:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.gelu.post", prs.after_gelu
        )
    return prs
