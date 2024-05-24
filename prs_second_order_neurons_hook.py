import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import argparse
import datetime
import json
from pathlib import Path


class SecondOrderPrsNeurons(object):
    def __init__(
        self,
        model,
        mlp_layer,
        device,
        coefficient: float = 100.
    ):
        self.mlp_layer = mlp_layer
        self.coefficient = coefficient
        self.model = model
        self.device = device
        self.reinit()

    def reinit(self):
        self.layer = 0
        # self.mlp_outputs = []
        self.post_gelu_outputs = None
        self.ln_1_mean = []
        self.attn_maps = []
        self.ln_1_std = []
        self.post_ln_mean = None
        self.post_ln_std = None
        self.finalized = False

    def log_attention_map(self, ret):
        self.attn_maps.append(ret.detach().cpu().numpy())  # [B, H, N, N]
        return ret
    
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu().numpy()  # [b, 1]
        return ret

    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu().numpy()  # [b, 1]
        return ret

    def log_ln_1_mean(self, ret):
        self.ln_1_mean.append(ret.detach().cpu().numpy())
        return ret

    def log_ln_1_std(self, ret):
        self.ln_1_std.append(ret.detach().cpu().numpy())
        return ret

    def log_post_gelu(self, ret):
        if self.mlp_layer == self.layer: 
            self.post_gelu_outputs = ret.detach().cpu().numpy()
        self.layer += 1
        return ret
    
    @torch.no_grad()
    def normalize_before_attention_per_neuron(self, input_tensor, layer_num):
        # Layer num is the output layer. input is [b, tokens, neurons, value]
        normalization_term = (1 + layer_num * 2) * input_tensor.shape[2]
        current_layer_norm = self.model.visual.transformer.resblocks[layer_num].ln_1
        mean_centered = input_tensor - self.ln_1_mean[:, layer_num, :, np.newaxis] / normalization_term
        weighted_mean_centered = (
            current_layer_norm.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.ln_1_std[:, layer_num, :, np.newaxis]
        bias_term = (
            current_layer_norm.bias.detach().to(self.device) / normalization_term
        )
        # Summing that at axis=2 gives us self.post_ln_1_norm[layer]
        return weighted_mean_by_std + bias_term

    @torch.no_grad()
    def apply_attention_matrix_per_neuron(self, input_tensor, layer_num, head_num):
        # input_tensor is [b, tokens, neurons, value] (x is the tokens)
        neurons = input_tensor.shape[2]
        attn_map = self.attn_maps[:, layer_num]  # [B, H, N, N]
        attn_layer = self.model.visual.transformer.resblocks[layer_num].attn
        num_heads = attn_layer.num_heads
        # get the current v's and bias
        _, _, v_weight = attn_layer._split_qkv_weight()
        _, _, v_bias = attn_layer._split_qkv_bias()
        v_bias = v_bias.detach().to(self.device)
        # v_bias.shape is [1, 16, 1, 64]
        v_weight = v_weight.detach().to(self.device)
        v = torch.einsum(
            "bnxc,dc->bnxd", input_tensor, v_weight.to(self.device)[head_num]
        ) 
        v = v + v_bias[:, head_num, :, :] / ((1 + layer_num * 2) * neurons) 
        x = torch.einsum("bnm,bmxc->bnxc", attn_map[:, head_num], v)
        ret = torch.einsum(
            "bnxc,dc->bnxd",
            x,
            attn_layer.out_proj.weight.detach()
            .to(self.device)
            .reshape(attn_layer.embed_dim, attn_layer.num_heads, attn_layer.head_dim)[
                :, head_num, :
            ],
        )
        ret += attn_layer.out_proj.bias.detach().to(self.device)[
            np.newaxis, np.newaxis, np.newaxis, :
        ] / ((1 + layer_num * 2) * num_heads * neurons)
        # ret is bnd. To get the projection to the class token we should do ret[:, 0]
        return ret

    @torch.no_grad()
    def project_attentions_to_output_per_neuron(self, bnmc, layer_in, layer_out):
        # [b, tokens, neurons, values]
        num_blocks = len(self.model.visual.transformer.resblocks)
        num_heads = self.model.visual.transformer.resblocks[layer_out].attn.num_heads
        normalization_term = (
            (1 + layer_out * 2) * (1 + num_blocks * 2) * num_heads * bnmc.shape[2]
        )
        # The first element is for how much each element before [out layer] is contributing
        # The second element is for how much out layer contributes to the output
        if layer_in >= layer_out:
            raise ValueError(
                "Wrong order: layer_in ({layer_in}) larger than layer_out ({layer_out})."
            )

        mean_centered = (
            bnmc[:, 0]
            - self.post_ln_mean.to(self.device)[:, np.newaxis] / normalization_term
        )  # [b c]

        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[:, np.newaxis]
        bias_term = (
            self.model.visual.ln_post.bias.detach().to(self.device) / normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)  # (b, 1024)

    def apply_mlp_post(self, mlps, layer):
        # mlps are [b, tokens, neurons]
        proj = self.model.visual.transformer.resblocks[layer].mlp.c_proj
        per_neuron_proj = torch.einsum('btn,nm->btnm', mlps, proj.weight.T)
        per_neuron_proj += proj.bias[np.newaxis, np.newaxis, np.newaxis, :] / mlps.shape[-1]
        return per_neuron_proj # [b, tokens, neurons, value]
        
    @torch.no_grad()
    def apply_attention_head_and_project_layer_neurons(self, layer, head, final_representation):
        # This includes everything but for one neuron layer, including projection to the output space.
        assert self.finalized
        neurons_per_layer = self.model.visual.transformer.resblocks[0].mlp_width
        assert self.mlp_layer < layer
        attentions = np.zeros(
            [
                self.post_gelu_outputs.shape[0],  # batch
                neurons_per_layer,
                self.model.visual.proj.shape[-1],
            ]
        )
        mlp_outputs = self.apply_mlp_post(
            self.post_gelu_outputs, self.mlp_layer
        ) # [b, tokens, neurons, value]
        normalized = self.normalize_before_attention_per_neuron(
            mlp_outputs, layer
        ) # [b, tokens, neurons, value]
        after_attention = self.apply_attention_matrix_per_neuron(
            normalized, layer, head
        ) # [b, tokens, neurons, values]
        projected = self.project_attentions_to_output_per_neuron(
            after_attention, self.mlp_layer, layer
        )
        attentions = projected.detach().cpu().numpy() * self.coefficient
        torch.cuda.empty_cache()
        # return attentions
        return (
            attentions
            / final_representation.norm(dim=-1)
            .detach()
            .cpu()
            .numpy()[:, np.newaxis, np.newaxis]
        )
    
    def finalize(self):
        if self.finalized:
            return
        # Now we can calculate the post-ln scaling, and project it (and normalize by the last norm)
        self.attn_maps = np.stack(self.attn_maps, axis=1)
        self.attn_maps = torch.from_numpy(self.attn_maps).to(self.device)
        self.post_gelu_outputs = torch.from_numpy(self.post_gelu_outputs).to(self.device)
        self.ln_1_mean = torch.from_numpy(np.stack(self.ln_1_mean, axis=1)).to(
            self.device
        )
        self.ln_1_std = torch.from_numpy(np.stack(self.ln_1_std, axis=1)).to(
            self.device
        )
        self.post_ln_mean = torch.from_numpy(self.post_ln_mean).to(self.device)
        self.post_ln_std = torch.from_numpy(self.post_ln_std).to(self.device)
        self.finalized = True


def hook_prs_logger(
    model,
    mlp_layer,
    device,
    coefficient=100,
):
    """Hooks a projected residual stream logger to the model."""
    prs = SecondOrderPrsNeurons(
        model,
        mlp_layer,
        device,
        coefficient=coefficient,
    )
    # model.hook_manager.register(
    #     "visual.transformer.resblocks.*.after_mlp", prs.log_mlps
    # )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.mlp.gelu.post", prs.log_post_gelu
    )
    model.hook_manager.register("visual.ln_post.mean", prs.log_post_ln_mean)
    model.hook_manager.register("visual.ln_post.sqrt_var", prs.log_post_ln_std)
    model.hook_manager.register(
        "visual.transformer.resblocks.*.ln_1.mean", prs.log_ln_1_mean
    )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.ln_1.sqrt_var", prs.log_ln_1_std
    )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.attn.attention.post_softmax",
        prs.log_attention_map,
    )
    return prs

