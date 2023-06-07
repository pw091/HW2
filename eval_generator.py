# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama.model_inference import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from llama.generation import LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    # torch.cuda.set_device(local_rank)
    # torch.set_default_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(checkpoint.keys())
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.FloatTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    evaluate: bool = False,
    hyper_param: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    prompts = [
        # Code related
        "function () => {\n x = x + 2 ret",
        "def sum(a, b): \n \t c = a + b",
        "<h>Heading1",
        "public static int sum(int a, int b) { \n \t return",
        # not code related,
        "Simply put graph theory, ",
        "My homework is going to be",
        # "I have two apples from my friend. What should I do if I'm hungry? "
    ]

    if hyper_param:
        f = open("/Users/anderson/Desktop/Project/LLaMA-From-Inference-to-Training/llama/history.json")
        history = json.load(f)
        output = open("result.txt", "a")
        for h in history:
            ckpt_dir = f"llama/ckpts/model_dim_{h['config']['dim']}_n_layers_{h['config']['n_layers']}_n_heads_{h['config']['n_heads']}_max_seq_len_{h['config']['max_seq_len']}"
            generator = load(
                ckpt_dir, tokenizer_path, local_rank, world_size, h["config"]["max_seq_len"], max_batch_size
            )
            output.write(f"config: dim_{h['config']['dim']}_n_layers_{h['config']['n_layers']}_n_heads_{h['config']['n_heads']}_max_seq_len_{h['config']['max_seq_len']}\n")
            output.write("==================================\n")
            results = generator.generate(prompts, max_gen_len=32, temperature=temperature, top_p=top_p)
            for result in results:
                output.write(f"Result: \n{result}\n")
                output.write("==================================\n")
    elif evaluate:
        output = open("eval.txt", "a")
        ckpt_dir = f"llama/ckpts/best_ckpt"
        output.write("==================================\n")
        generator = load(
            ckpt_dir, tokenizer_path, local_rank, world_size, 512, max_batch_size
        )
        results = generator.generate(prompts, max_gen_len=32, temperature=temperature, top_p=top_p)
        for i, r in enumerate(results):
            output.write(f"Prompt: \n{prompts[i]}\n")
            output.write("==================================\n")
            output.write(f"Result: \n{r}\n")
            output.write("==================================\n")
    else:
        generator = load(
            ckpt_dir, tokenizer_path, local_rank, world_size, 512, max_batch_size
        )
        results = generator.generate(prompts, max_gen_len=32, temperature=temperature, top_p=top_p)
        for i, r in enumerate(results):
            print(f"Prompt: \n{prompts[i]}\n")
            print("==================================\n")
            print(f"Result: \n{r}\n")
            print("==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
