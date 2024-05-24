import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from torchvision.datasets import ImageNet, ImageFolder, CIFAR10, CIFAR100
from utils.factory import create_model_and_transforms, get_tokenizer
import torch.multiprocessing
from utils.subsampled_imagenet import SubsampledImageNet


def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="openai", type=str)
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/datasets/ilsvrc/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--coefficient",
        default=100.0,
        type=float,
        help="We multiply all the representations so numbers will be larger. It does not change anything",
    )
    parser.add_argument(
        "--split", default="train", type=str, help="test, valid, or train."
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save",
    )
    parser.add_argument("--dataset", type=str, default="imagenet", help="")
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""

    torch.multiprocessing.set_sharing_strategy("file_system")
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

    dataset = {
        "imagenet": SubsampledImageNet,
        "CIFAR100": CIFAR100,
        "CIFAR10": CIFAR10,
    }[args.dataset](args.data_path, split=args.split, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    representation_results = []
    classes_results = []
    for i, (p, classes) in enumerate(tqdm.tqdm(dataloader)):
        image = p.to(args.device)
        with torch.no_grad():
            classes_results.append(classes.detach().cpu().numpy())
            representation = model.encode_image(
                image.to(args.device), attn_method="head", normalize=True
            )
            representation_results.append(
                args.coefficient * representation.detach().cpu().numpy()
            )

    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.split}_representations_{args.model}_{args.pretrained}.npy",
        ),
        "wb",
    ) as f:
        np.save(f, np.concatenate(representation_results, axis=0))
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_{args.split}_classes.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(classes_results, axis=0))


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
