import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import argparse
import numpy as np
import json
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from PIL import Image
import os.path
import argparse
from pathlib import Path
import random
from utils.factory import create_model_and_transforms, get_tokenizer
from sentence_transformers import SentenceTransformer, util


REQUEST = """I want to generate an image by providing image descriptions as input to a text-to-image model.  
Each of the image descriptions must include the word "{{yes}}". 
They must not include the word "{{no}}", any synonym of it or plural version!
The image descriptions should include as many words as possible from the next list and almost no other words:

{{list}}

Do not use names of people or places from the list unless they are famous and there is something visually distinctive about them. 
In each of the image descriptions mention as many objects and animals as possible from the list above. 
If you want to mention the place in which the image is taken or a name of a person, describe it with visually distinctive words. 
For example, if "paris" is in the list, instead of saying "... in Paris", say "... with the Eiffel Tower in the background" or "... next to a sign saying 'Paris'".
Don't mention words that are visually similar to "{{no}}", even if they are in the list above. 
For example, if the word was "tree" you should not mention "trees", "leafs" or "eucalyptus" and if the word was "board" you should not mention "blackboard", "whiteboard" or "writing area". 
Only use words that you know what they mean. 
Generate a list of 50 image descriptions.
"""


def get_args_parser():
    parser = argparse.ArgumentParser("Combine to sentences", add_help=False)

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
    parser.add_argument(
        "--class_0", default="dog", type=str
    )  # The text should not have that
    parser.add_argument(
        "--class_1", default="cat", type=str
    )  # The text should have that
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument("--dataset_path", default="./dataset", type=str)
    parser.add_argument("--dataset", type=str, default="imagenet", help="")
    parser.add_argument("--mlp_layers", default=[8, 9, 10], nargs="+", type=int)
    parser.add_argument("--top_k_pca", default=100, type=int)
    parser.add_argument("--coefficient", default=100.0, type=float)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument("--input_file", default="", help="path where data is saved")
    parser.add_argument(
        "--descriptions_dir",
        default="./text_descriptions",
        help="path where data is saved",
    )
    parser.add_argument(
        "--text_descriptions",
        default="30k",
        help="Names of the text descriptions",
    )

    parser.add_argument(
        "--cosim_similarity", default=0.6, help="text similarity to remove synonyms."
    )
    parser.add_argument(
        "--neurons_num", default=50, type=int, help="how many neurons to take"
    )
    parser.add_argument(
        "--results_per_generation",
        default=10,
        type=int,
        help="how many time to generate",
    )

    parser.add_argument(
        "--neurons_texts",
        default=128,
        type=int,
        help="how many words per neurons to take",
    )
    parser.add_argument(
        "--overall_words", default=20, type=int, help="how many words to take overall"
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help='Image generations per query'
    )
    return parser


def get_text_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def get_image_model(device):
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
    ).to(device)
    stage_1.enable_model_cpu_offload()

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    stage_2.enable_model_cpu_offload()
    return stage_1, stage_2


def send_request(positive, yes, no, model, tokenizer):
    request = (
        REQUEST.replace("{{list}}", "[" + ", ".join(sorted(set(positive))) + "]")
        .replace("{{yes}}", yes)
        .replace("{{no}}", no)
    )
    print(request)
    messages = [
        {
            "role": "system",
            "content": "You are a capable instruction-following AI agent.",
        },
        {"role": "user", "content": request},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    return tokenizer.decode(response, skip_special_tokens=True)


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    block_list = {
        "automobile": ["tesla", "automobile", "car", "sebring"],
        "truck": ["van", "silverado", "truck", "hummer"],
        "bird": ["turkey", "bird", "penguin", "eagle"],
        "frog": ["frog"],
        "horse": [
            "horse",
            "pony",
            "ponies",
        ],
        "deer": ["deer"],
        "cat": ["cat"],
        "ship": ["ship", "boat"],
        "dog": ["puppies", "pug", "puppy", "dog"],
        "airplane": ["airplane"],
        "green light": ["green light", "emerald"],
        "stop sign": ["stop sign"],
        "crossroad": ["crossroad"],
        "yield": ["yield"],
        "limit 25": ["limit 25"],
        "vacuum cleaner": ["vacuum cleaner", "vacuum", "vacuums"],
    }
    generator = torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    text_descriptions = np.load(
        os.path.join(
            args.output_dir,
            f"{args.text_descriptions}_{args.model}_{args.pretrained}.npy",
        )
    )
    text_descriptions = text_descriptions - text_descriptions.mean(axis=0)
    text_descriptions = (
        args.coefficient
        * text_descriptions
        / np.linalg.norm(text_descriptions, axis=0, keepdims=True)
    )

    # Load all the pcas
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
    # Load all the norms:
    tokenizer = get_tokenizer(args.model)
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained, force_quick_gelu=True
    )
    model.eval().to(args.device)
    texts = tokenizer([args.class_0, args.class_1]).to(args.device)  # tokenize
    class_embeddings = model.encode_text(texts, normalize=True)
    class_embeddings = class_embeddings.detach().cpu().numpy()
    class_embeddings_direction = class_embeddings[0] - class_embeddings[1]
    max_directions = pcas @ class_embeddings_direction
    most_significant_neurons = np.argsort(np.abs(max_directions))

    all_words = []
    loaded_dict = {}
    index = -1
    for mlp_layer in args.mlp_layers:
        path = f"{args.dataset}_mlps_{args.model}_{args.pretrained}_{mlp_layer}_{args.text_descriptions}_decomposition_omp_1.0_{args.neurons_texts}.json"
        current_loaded_dict = json.load(
            open(
                os.path.join(args.output_dir, path),
                "r",
            )
        )
        current_index = -1
        for key, value in current_loaded_dict.items():
            loaded_dict[str(1 + index + int(key))] = value
            current_index = max(current_index, 1 + index + int(key))
        index = current_index
    for neuron in most_significant_neurons[-args.neurons_num :]:
        all_words += [
            (i[1] * max_directions[neuron], i[2]) for i in loaded_dict[str(neuron)]
        ]
    sim_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    class_0_rep = sim_model.encode([args.class_0], convert_to_tensor=True)[0]
    new_all_words = []
    for word in all_words:
        cosim = (
            util.cos_sim(
                sim_model.encode([word[1].strip()], convert_to_tensor=True)[0],
                class_0_rep,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        if (
            cosim < args.cosim_similarity
            and word[1].strip() not in block_list[args.class_0]
            and args.class_0 != word[1].strip()
        ):
            new_all_words.append(word)
        else:
            print(f"Removed {word}")
    all_words = [i[1].strip() for i in sorted(new_all_words)[-args.overall_words :]]

    print(all_words)
    all_words = list(set(all_words))
    text_model, text_tokenizer = get_text_model()  # Text
    results = send_request(
        [i.strip() for i in all_words],
        args.class_1,
        args.class_0,
        text_model,
        text_tokenizer,
    )
    results_tmp = results.split("\n")
    results = []
    add_res = False
    for i in results_tmp:
        if i.startswith("1") and len(i) > 1:
            add_res = True
        if add_res:
            results.append(i)
    print(results)
    del text_model
    del text_tokenizer
    torch.cuda.empty_cache()
    # Do the diffusion part:
    stage_1, stage_2 = get_image_model(args.device)
    outputs = []
    avg_results = 0
    good_results = []
    new_results = []
    for r in results:
        if args.class_0 not in r.lower() and args.class_1 in r.lower():
            new_results.append(r)
    results = results[:50]
    counter = 0
    random.shuffle(results)
    if len(new_results) >= args.results_per_generation:
        new_results = new_results[: args.results_per_generation]
    else:
        raise ValueError("Not working!")
    for r in new_results:
        if r and r[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            r = r[r.index(" ") + 1 :]
        if not r:
            continue
        # text embeds
        prompt_embeds, negative_embeds = stage_1.encode_prompt(
            r, device=args.device, num_images_per_prompt=args.batch_size
        )
        # stage 1
        image = stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images
        # stage 2
        image = stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
            guidance_scale=4.0,
        ).images
        for x in range(args.batch_size):
            pt_to_pil(image)[x].save(
                os.path.join(args.dataset_path, f"{counter:03}.png")
            )

            image_pil = Image.open(os.path.join(args.dataset_path, f"{counter:03}.png"))
            image_processed = preprocess(image_pil)[np.newaxis, :, :, :].to(args.device)
            repp = model.encode_image(image_processed).detach().cpu().numpy()
            classification_result = repp[0] @ class_embeddings_direction
            if classification_result > 0:
                good_results.append(counter)
            outputs.append(
                (
                    f"{counter:03}.png",
                    r,
                    args.class_0,
                    args.class_1,
                    str(classification_result),
                )
            )
            avg_results += classification_result
            counter += 1
    with open(os.path.join(args.dataset_path, f"files.csv"), "w") as f:
        for r in outputs:
            f.write("\t".join(r) + "\n")
    print("Successful images:", good_results)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset_path:
        Path(args.dataset_path).mkdir(parents=True, exist_ok=True)
    main(args)
