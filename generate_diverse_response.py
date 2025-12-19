import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import re
from PIL import Image
import datetime
import random
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def set_seed(seed: int):
    """
    Set the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_response(model, tokenizer, image_processor, prompt, image, args):
    """
    Generates a response from the model given the prompt and image.
    """
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return output


def run_experiment(args):
    set_seed(42)

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    # Determine conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    with open(args.dataset_file, "r") as f:
        dataset = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    args.output_file = os.path.join(args.output_dir, f"results_{formatted_time}.jsonl")


    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            image = data["image"]
            #id = data["id"]

            image_path = f"{args.image_dir}/{image}"
            image = Image.open(image_path).convert("RGB")

            outputs = []
            prompts = [
                "Describe the following image in detail",
                "Provide a detailed description of the given image",
                "Give an elaborate explanation of the image you see",
                "Share a comprehensive rundown of the presented image",
                "Offer a thorough analysis of the image",
                "Explain the various aspects of the image before you",
                "Clarify the contents of the displayed image with great detail",
                "Characterize the image using a well-detailed description",
                "Break down the elements of the image in a detailed manner",
                "Walk through the important details of the image",
                "Portray the image with a rich, descriptive narrative",
                "Narrate the contents of the image with precision",
                "Analyze the image in a comprehensive and detailed manner",
                "Illustrate the image through a descriptive explanation",
                "Examine the image closely and share its details",
                "Write an exhaustive depiction of the given image"
            ]
            for i in range(args.num_responses):
                #qs = data["conversations"][0]["value"].replace("<image>", "").strip()
                qs = random.choice(prompts)
                # Add special tokens
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                output = generate_response(model, tokenizer, image_processor, prompt, image, args)
                outputs.append(output)


            with open(args.output_file, "a") as f:
                f.write(json.dumps({**data, "outputs": outputs}) + "\n")

        except Exception as e:
            print(e)

    args.output_file = args.output_file.replace(".jsonl", "_args.json")
    args_dict = vars(args)
    with open(args.output_file, "w") as f:
        json.dump({**args_dict}, f, indent=4)


def main(args):
    run_experiment(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default="liuhaotian/llava-v1.5-7b", required=True, help='Path to the model')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-file", type=str, default="/home/vqa/data/dataset")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default="/home/vqa/data/dataset")
    parser.add_argument('--conv-mode', type=str, default=None, help='Conversation mode')
    parser.add_argument('--sep', type=str, default=',', help='Separator')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p for nucleus sampling')
    parser.add_argument('--num-beams', type=int, default=1, help='Number of beams for beam search')
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Maximum number of new tokens to generate')
    parser.add_argument('--num-responses', type=int, default=1, help='Number of responses to generate for each prompt')


    args = parser.parse_args()

    main(args)
