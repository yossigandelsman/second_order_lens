import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms, get_tokenizer
from prs_second_order_neurons_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder
import scipy as sp
from utils.subsampled_imagenet import SubsampledImageNet


def get_args_parser():
    parser = argparse.ArgumentParser("Project Second Order Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--mlp_layer", default=9, type=int)
    parser.add_argument("--pretrained", default="openai", type=str)
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets/ilsvrc/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="datasets"
    )
    parser.add_argument("--split", type=str, default="train", help="val or train")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, force_quick_gelu=True
    )
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.mlp_layer, args.device)
    if args.dataset == "imagenet":
        ds = SubsampledImageNet(
            root=args.data_path, split=args.split, transform=preprocess
        )
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        raise NotImplementedError("Add other datasets here")
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    attn_layer = model.visual.transformer.resblocks[0].attn
    num_heads = attn_layer.num_heads
    mlp_results = []
    for i, (image, _) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            prs.reinit()
            representation = model.encode_image(
                image.to(args.device), attn_method="head", normalize=False
            )
            prs.finalize()
            current_res = []
            for layer in range(
                args.mlp_layer + 1, len(model.visual.transformer.resblocks)
            ):
                for head in range(num_heads):
                    current_res.append(
                        prs.apply_attention_head_and_project_layer_neurons(
                            layer, head, representation
                        )
                    )
            mlp_results.append(np.stack(current_res, axis=0).sum(axis=0))

    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.split}_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_merged.npy",
        ),
        "wb",
    ) as f:
        np.save(f, np.concatenate(mlp_results, axis=0))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
