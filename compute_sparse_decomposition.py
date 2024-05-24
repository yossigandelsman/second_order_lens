import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import time
from utils.misc import accuracy
from torch.utils.data import DataLoader
import tqdm
import scipy as sp
from sklearn.decomposition import SparseCoder
from utils.factory import create_model_and_transforms, get_tokenizer
from prs_second_order_neurons_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder
from decompose import Decompose
from utils.subsampled_imagenet import SubsampledValImageNet
import json


def get_args_parser():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser("Sparse decomposition part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets/ilsvrc/",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--coefficient", default=100.0, type=float)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--mlp_layer", default=9, type=int)
    parser.add_argument("--components", default=32, type=int)
    parser.add_argument("--top_k_pca", default=100, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")

    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--descriptions_dir",
        default="./text_descriptions",
        help="path where data is saved",
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--text_descriptions",
        default="30k",
        help="Names of the text descriptions",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="imagenet, waterbirds, cub, binary_waterbirds",
    )
    parser.add_argument(
        "--transform_alpha", type=float, default=1.0, help="The algorithm alpha"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the reconstruction")
    parser.set_defaults(evaluate=False)
    return parser


def main(args):
    classifier = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_classifier_{args.model}_{args.pretrained}.npy",
        ),
        mmap_mode="r",
    )
    text_descriptions = np.load(
        os.path.join(
            args.input_dir,
            f"{args.text_descriptions}_{args.model}_{args.pretrained}.npy",
        ),
        mmap_mode="r",
    )
    text_descriptions = text_descriptions - text_descriptions.mean(axis=0)
    text_descriptions = (
        args.coefficient
        * text_descriptions
        / np.linalg.norm(text_descriptions, axis=0, keepdims=True)
    )
    print('Loaded texts')
    neurons = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_merged.npy",
        ),
        mmap_mode="r",
    )
    print('Loaded neurons')
    neurons_mean = neurons.mean(axis=0)

    pcas = np.load(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.top_k_pca}_pca.npy",
        ),
        mmap_mode='r'
    )  # [neurons, d]
    assert len(neurons_mean.shape) == 2, neurons_mean.shape
    print("Computing sparse dictionary")
    before = time.time()
    coder = Decompose(
        text_descriptions,
        l1_penalty=args.transform_alpha,
        transform_n_nonzero_coefs=args.components,
    )
    decomposition = coder.transform(pcas)  # [neurons, text_descriptions]
    print("Done in", (time.time() - before) / (60 * 60), "hours")
    with open(
        os.path.join(args.descriptions_dir, f"{args.text_descriptions}.txt"), "r"
    ) as f:
        lines = [i.replace("\n", "") for i in f.readlines()]
    sparse_decomposition = decomposition.copy()
    name = f"{args.dataset}_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.text_descriptions}_decomposition_omp_{args.transform_alpha}_{args.components}"
    jsn = {}
    for neuron in tqdm.trange(decomposition.shape[0]):
        jsn[neuron] = []
        all_addresses = np.argsort(np.abs(decomposition[neuron]))
        addresses = all_addresses[-args.components :]
        zero_address = all_addresses[: -args.components]
        for j in addresses[::-1]:
            jsn[neuron].append((int(j), float(decomposition[neuron, j]), str(lines[j])))
        sparse_decomposition[neuron, zero_address] = 0
    with open(
        os.path.join(
            args.output_dir,
            f"{name}.json",
        ),
        "w",
    ) as f:
        json.dump(jsn, f)
    # Save the decomposition
    sparse_matrix = sp.sparse.csc_matrix(decomposition)

    sp.sparse.save_npz(
        os.path.join(
            args.output_dir,
            f"{name}.npz",
        ),
        sparse_matrix,
    )
    if not args.evaluate:
        return 
    #### This the evaluation part, done on the training data
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, force_quick_gelu=True
    )
    model.to(args.device)
    model.eval()
    prs = hook_prs_logger(
        model, args.mlp_layer, args.device, coefficient=args.coefficient
    )
    
    reconstruction = sparse_decomposition @ text_descriptions  # [neurons, repr]
    ds = SubsampledValImageNet(root=args.data_path, split=args.split, transform=preprocess)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    attn_layer = model.visual.transformer.resblocks[0].attn
    num_heads = attn_layer.num_heads
    ablated_representation = []
    all_cls = []
    for i, (image, classes) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            all_cls.append(classes.detach().cpu().numpy())
            prs.reinit()
            representation_nonorm = model.encode_image(
                image.to(args.device), attn_method="head", normalize=False
            )
            prs.finalize()
            representation = args.coefficient * (
                representation_nonorm
                / representation_nonorm.norm(dim=-1)[:, np.newaxis]
            )
            current_res = []
            for layer in range(
                args.mlp_layer + 1, len(model.visual.transformer.resblocks)
            ):
                for head in range(num_heads):
                    current_res.append(
                        prs.apply_attention_head_and_project_layer_neurons(
                            layer, head, representation_nonorm
                        )
                    )  # (B, 4096, 768)
            mlp_results = np.stack(current_res, axis=0).sum(axis=0) # [b, n, d]
            representation_mean_ablated = (
                representation.detach().cpu().numpy()
                - mlp_results.sum(axis=1)
                + neurons_mean.sum(axis=0)[np.newaxis, :]
            )
            mlp_results_mean_ablated = (
                mlp_results - neurons_mean[np.newaxis, :, :]
            )  # [b, neuron, d]
            assert len(mlp_results_mean_ablated.shape) == 3, mlp_results_mean_ablated.shape
            assert len(reconstruction.shape) == 2, reconstruction.shape
            coefs = np.einsum("bnd,nd->bn", mlp_results_mean_ablated, reconstruction)[:, :, np.newaxis]
            representation_mean_ablated  += (
                coefs * reconstruction[np.newaxis, :, :]
            ).sum(axis=1)
            ablated_representation.append(
                representation_mean_ablated @ classifier
            )
    all_cls = np.concatenate(all_cls, axis=0)
    ablated_representation = np.concatenate(ablated_representation, axis=0)
    ablated_representation_acc = (
        accuracy(
            torch.from_numpy(ablated_representation).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("Reconstructed neurons:", ablated_representation_acc)            


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
