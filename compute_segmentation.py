import argparse
import torch
import numpy as np
import scipy
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import imageio
import cv2
import os
from pathlib import Path
import tqdm
from utils.factory import create_model_and_transforms
from utils.imagenet_segmentation import ImagenetSegmentation
from utils.segmentation_utils import (
    batch_pix_accuracy,
    batch_intersection_union,
    get_ap_scores,
    Saver,
)
from sklearn.metrics import precision_recall_curve
from prs_neuron_record import hook_switch_logger


# Args
def get_args_parser():
    parser = argparse.ArgumentParser(description="Segmentation scores")
    parser.add_argument("--save_img", action="store_true", default=False, help="")
    parser.add_argument("--random", action="store_true", default=False, help="")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="imagenet_seg",
        help="The name of the dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="The name of the classifier dataset",
    )
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument("--thr", type=float, default=0.0, help="threshold")
    parser.add_argument(
        "--data_path",
        default="imagenet_seg/gtsegs_ijcv.mat",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--classifier_dir", default="./output_dir/")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="openai", type=str)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--mlp_layers", default=[8, 9, 10], nargs="+", type=int)
    parser.add_argument("--top_k_pca", default=100, type=int)
    parser.add_argument("--top_neurons", default=200, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def eval_batch(model, prs, image, labels, index, args, max_per_class, classifier, saver):
    # max_per_class is [1000, neurons, 2]
    # Save input image
    if args.save_img:
        # Saves one image from each batch
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype("uint8")
        Image.fromarray(img, "RGB").save(
            os.path.join(saver.results_dir, "input/{}_input.png".format(index))
        )
        Image.fromarray(
            (labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype(
                "uint8"
            ),
            "RGB",
        ).save(os.path.join(saver.results_dir, "input/{}_mask.png".format(index)))

    # Get the model attention maps:
    prs.layers = []
    prs.indices = []
    prs.reinit()
    representation = model.encode_image(
        image.to(args.device), normalize=False
    )
    chosen_class = (representation.detach().cpu().numpy() @ classifier).argmax(axis=1)
    prs.layers = [i[1] for i in max_per_class[chosen_class[0]]]
    prs.indices = [i[0] for i in max_per_class[chosen_class[0]]]
    prs.reinit()
    representation = model.encode_image(
        image.to(args.device), normalize=False
    )
    
    patches = args.image_size // model.visual.patch_size[0]
    results = torch.from_numpy(np.mean(np.stack(prs.recorded, axis=0), axis=0)[:, 1:].reshape(1, patches, patches))

    Res = torch.nn.functional.interpolate(
        results[:, np.newaxis], scale_factor=model.visual.patch_size[0], mode="bilinear"
    ).to(args.device)
    # Res = torch.clip(Res, 0, Res.max())
    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = 0.5 #  Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [224, 224], mode="bilinear")
        mask = mask[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype("uint8")
        imageio.imsave(
            os.path.join(args.exp_img_path, "mask_" + str(index) + ".jpg"), mask
        )

        relevance = F.interpolate(Res, [224, 224], mode="bicubic")
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        hm = np.clip(255.0 * hm / hm.max(), 0, 255.0).astype(np.uint8)
        high = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        imageio.imsave(
            os.path.join(args.exp_img_path, "heatmap_" + str(index) + ".jpg"), high
        )

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap = 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    batch_ap += ap

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, pred, target


def _create_saver_and_folders(args):
    saver = Saver(args)
    saver.results_dir = os.path.join(saver.experiment_dir, "results")
    if not os.path.exists(saver.results_dir):
        os.makedirs(saver.results_dir)
    if not os.path.exists(os.path.join(saver.results_dir, "input")):
        os.makedirs(os.path.join(saver.results_dir, "input"))
    if not os.path.exists(os.path.join(saver.results_dir, "explain")):
        os.makedirs(os.path.join(saver.results_dir, "explain"))

    args.exp_img_path = os.path.join(saver.results_dir, "explain/img")
    if not os.path.exists(args.exp_img_path):
        os.makedirs(args.exp_img_path)
    return saver


def main(args):
    # Model
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, force_quick_gelu=True
    )
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size
    num_of_neurons = model.visual.transformer.resblocks[0].mlp_width
    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))
    pcas = []
    for mlp_layer in args.mlp_layers:
        curr_pcas = np.load(
            os.path.join(
                args.output_dir,
                f"{args.dataset}_train_mlps_{args.model}_{args.pretrained}_{mlp_layer}_{args.top_k_pca}_pca.npy",
            ),
            mmap_mode="r",
        )  # [neurons, d]
        pcas.append(curr_pcas)
    pcas = np.concatenate(pcas)
    
    prs = hook_switch_logger(model, device=args.device, layers=[], indices=[], post_gelu=True)
    # Data
    target_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), Image.NEAREST),
        ]
    )

    ds = ImagenetSegmentation(
        args.data_path, transform=preprocess, target_transform=target_transform
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    iterator = tqdm.tqdm(dl)
    # Saver
    saver = _create_saver_and_folders(args)
    # Classifier
    with open(
        os.path.join(
            args.classifier_dir,
            f"{args.dataset}_classifier_{args.model}_{args.pretrained}.npy",
        ),
        "rb",
    ) as f:
        classifier = np.load(f)
    # Eval in loop
    max_directions = pcas @ classifier # (3 * 3072) x 1000
    max_per_class = []
    if not args.random:
        for i in range(max_directions.shape[1]):
            max_per_class.append(np.argsort(np.abs(max_directions[:, i]))[-args.top_neurons:])
    else:
        for i in range(max_directions.shape[1]):
            max_per_class.append(np.random.choice(max_directions.shape[0], args.top_neurons, replace=False))
    
    max_per_class = [[(i % num_of_neurons, i // num_of_neurons + args.mlp_layers[0]) for i in x] for x in max_per_class]
    total_inter, total_union, total_correct, total_label = (
        np.int64(0),
        np.int64(0),
        np.int64(0),
        np.int64(0),
    )
    total_ap = []

    predictions, targets = [], []
    for batch_idx, (image, labels) in enumerate(iterator):

        images = image.to(args.device)
        labels = labels.to(args.device)

        correct, labeled, inter, union, ap, pred, target = eval_batch(
            model, prs, images, labels, batch_idx, args, max_per_class, classifier, saver
        )

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype("int64")
        total_label += labeled.astype("int64")
        total_inter += inter.astype("int64")
        total_union += union.astype("int64")
        total_ap += [ap]
        pixAcc = (
            np.float64(1.0)
            * total_correct
            / (np.spacing(1, dtype=np.float64) + total_label)
        )
        IoU = (
            np.float64(1.0)
            * total_inter
            / (np.spacing(1, dtype=np.float64) + total_union)
        )
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        iterator.set_description(
            "pixAcc: %.4f, mIoU: %.4f, mAP: %.4f" % (pixAcc, mIoU, mAp)
        )

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    pr, rc, thr = precision_recall_curve(targets, predictions)
    np.save(os.path.join(saver.experiment_dir, "precision.npy"), pr)
    np.save(os.path.join(saver.experiment_dir, "recall.npy"), rc)

    txtfile = os.path.join(saver.experiment_dir, "result_mIoU_%.4f.txt" % mIoU)
    fh = open(txtfile, "w")
    print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
    print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
    print("Mean AP over %d classes: %.4f\n" % (2, mAp))

    fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
    fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
    fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
    fh.close()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
