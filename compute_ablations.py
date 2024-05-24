import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path
import copy
import tqdm
from utils.misc import accuracy
from torch.utils.data import DataLoader
import tqdm
from utils.factory import create_model_and_transforms
from prs_second_order_neurons_hook import hook_prs_logger
from torchvision.datasets import ImageNet
import torch.multiprocessing
from utils.subsampled_imagenet import SubsampledValImageNet


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument("--mlp_layer", default=9, type=int)
    parser.add_argument("--coefficient", default=100.0, type=float)
    parser.add_argument("--top_k_pca", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir/", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir",
        default="./output_dir/",
        help="path where data is saved",
    )
    parser.add_argument(
        "--data_path",
        default="/datasets/ilsvrc/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--split", default="val", type=str, help="test, valid, or train."
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--dataset", type=str, default="imagenet", help="")
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    classifier = np.load(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_classifier_{args.model}_{args.pretrained}.npy",
        )
    )
    classifier = torch.from_numpy(classifier).float().to(args.device)
    neurons = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_merged.npy",
        ),
        mmap_mode="r",
    )  # [images, neurons, d]
    neurons_mean = neurons.mean(axis=0)
    # Get pcas
    pcas = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.top_k_pca}_pca.npy",
        ),
        mmap_mode="r",
    )
    norms = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.top_k_pca}_norm.npy",
        ),
        mmap_mode="r",
    )
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, force_quick_gelu=True
    )
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size
    prs = hook_prs_logger(
        model, args.mlp_layer, args.device, coefficient=args.coefficient
    )
    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))
    
    ds = SubsampledValImageNet(root=args.data_path, split=args.split, transform=preprocess)
    print('Please note that we evaluate on 10% of ImageNet validation set. Therefore, the results are slightly different from the paper.')
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    attn_layer = model.visual.transformer.resblocks[0].attn
    num_heads = attn_layer.num_heads
    all_cls = []
    original_results = []
    without_neurons = []
    without_significant = []
    without_insignificant = []
    projected_to_pc = []
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
            original_results.append(
                (representation @ classifier).detach().cpu().numpy()
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
                    )
            mlp_results = np.stack(current_res, axis=0).sum(axis=0)
            mlp_results_mean_ablated = (
                mlp_results - neurons_mean[np.newaxis, :, :]
            )  # [b, neuron, d]
            mlp_results_norm = np.linalg.norm(
                mlp_results_mean_ablated, axis=-1
            )  # [b, neurons]
            representation_wo_sig = (
                copy.deepcopy(representation.detach().cpu().numpy())
                - mlp_results.sum(axis=1)
                + neurons_mean.sum(axis=0)[np.newaxis, :]
            )
            representation_wo_insig = copy.deepcopy(representation_wo_sig)
            representation_pc = copy.deepcopy(representation_wo_sig)
            without_neurons.append(
                representation_wo_sig @ classifier.detach().cpu().numpy()
            )  # This is just mean ablation
            for batch in range(image.shape[0]):
                indices = mlp_results_norm[batch] > norms[:, 0] # We want the threshold, not all the top norms, so [:, 0]
                sig_sum = mlp_results_mean_ablated[batch][indices].sum(axis=0)
                insig_sum = mlp_results_mean_ablated[batch][
                    np.logical_not(indices)
                ].sum(axis=0)
                representation_wo_sig[batch] += insig_sum
                representation_wo_insig[batch] += sig_sum
                # Projection : (np.dot(a, b) / np.linalg.norm(b)**2 ) * b
                representation_pc[batch] += (
                    np.einsum("nd,nd->n", mlp_results_mean_ablated[batch], pcas)[
                        :, np.newaxis
                    ]
                    * pcas
                ).sum(axis=0)
            without_significant.append(
                representation_wo_sig @ classifier.detach().cpu().numpy()
            )
            without_insignificant.append(
                representation_wo_insig @ classifier.detach().cpu().numpy()
            )
            projected_to_pc.append(
                representation_pc @ classifier.detach().cpu().numpy()
            )

    all_cls = np.concatenate(all_cls, axis=0)
    original_results = np.concatenate(original_results, axis=0)
    without_neurons = np.concatenate(without_neurons, axis=0)
    without_significant = np.concatenate(without_significant, axis=0)
    without_insignificant = np.concatenate(without_insignificant, axis=0)
    projected_to_pc = np.concatenate(projected_to_pc, axis=0)
    original_acc = (
        accuracy(
            torch.from_numpy(original_results).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("Baseline (from representations):", original_acc)

    without_neurons_acc = (
        accuracy(
            torch.from_numpy(without_neurons).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("Without all the neurons:", without_neurons_acc)

    without_significant_neurons_acc = (
        accuracy(
            torch.from_numpy(without_significant).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("Without significant neurons:", without_significant_neurons_acc)

    without_insignificant_neurons_acc = (
        accuracy(
            torch.from_numpy(without_insignificant).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("Without insignificant neurons:", without_insignificant_neurons_acc)

    projected_to_pc_acc = (
        accuracy(
            torch.from_numpy(projected_to_pc).float(),
            torch.from_numpy(all_cls),
        )[0]
        * 100
    )
    print("With neurons reconstructed from PC:", projected_to_pc_acc)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
