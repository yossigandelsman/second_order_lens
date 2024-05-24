import numpy as np
import os.path
import argparse
from pathlib import Path
import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("Compute PCAs", add_help=False)

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
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir",
        default="./output_dir/",
        help="path where data is saved",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--dataset", type=str, default="imagenet", help="")
    parser.add_argument("--mlp_layer", default=9, type=int)
    parser.add_argument("--top_k_pca", default=100, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    neurons = np.load(
        os.path.join(
            args.input_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_merged.npy",
        ),
        mmap_mode="r",
    )  # [images, neurons, d]
    neurons_mean = neurons.mean(axis=0)
    # Compute the pcas
    pcas = []
    norms = []
    for neuron in tqdm.trange(neurons.shape[1]):
        current_neurons = neurons[:, neuron] - neurons_mean[neuron]
        important = np.argsort(np.linalg.norm(current_neurons, axis=-1))[
            -args.top_k_pca :
        ]
        current_important_neurons = current_neurons[important]
        norms.append(
            np.sort(np.linalg.norm(current_neurons, axis=-1))[-args.top_k_pca :]
        )
        u, s, vh = np.linalg.svd(
            current_important_neurons, full_matrices=False
        )  # (u * s) @ vh is value
        how_many_positive = len(
            np.nonzero(1.0 * ((current_important_neurons @ vh[0]) > 0))[0]
        )
        # Set the direction:
        if how_many_positive > args.top_k_pca // 2:
            pcas.append(vh[0])
        else:
            pcas.append(-vh[0])
    pcas = np.stack(pcas, axis=0)  # [neurons, d]
    norms = np.stack(norms, axis=0)  # [neurons]
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.top_k_pca}_pca.npy",
        ),
        "wb",
    ) as w:
        np.save(w, pcas)
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{args.mlp_layer}_{args.top_k_pca}_norm.npy",
        ),
        "wb",
    ) as w:
        np.save(w, norms)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
