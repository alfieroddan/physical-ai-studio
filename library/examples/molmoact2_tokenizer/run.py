# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load Qwen2Tokenizer from a bundled MolmoAct2 tokenizer JSON file."""

from pathlib import Path

from transformers import Qwen2Tokenizer

IMAGE_TOKEN = "<|image|>"
IMAGE_TOKEN_ID = 154629
BOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
MAX_LENGTH = 64


def _tokenizer_options() -> dict[str, object]:
    return {
        "bos_token": BOS_TOKEN,
        "eos_token": BOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "model_max_length": MAX_LENGTH,
        "padding_side": "right",
    }


def main() -> None:
    """Load the bundled tokenizer and encode representative MolmoAct2 prompts."""
    tokenizer_dir = Path(__file__).parent
    tokenizer = Qwen2Tokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        **_tokenizer_options(),
    )

    prompts = [
        f"{IMAGE_TOKEN}<|im_start|>user\nPick up the red block.<|im_end|>\n",
        f"Image 1{IMAGE_TOKEN}<|im_start|>user\nMove left, then close the gripper.<|im_end|>",
        "tabletop joint",
        "Unicode check: café 日本語",
    ]
    encode_options = {
        "add_special_tokens": False,
        "max_length": MAX_LENGTH,
        "padding": "max_length",
        "truncation": True,
    }
    output = tokenizer(prompts, **encode_options)

    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_TOKEN_ID:
        msg = f"Expected {IMAGE_TOKEN!r} to have ID {IMAGE_TOKEN_ID}"
        raise RuntimeError(msg)

    print(f"tokenizer: {type(tokenizer).__name__}")
    print(f"input_ids: {output['input_ids']}")
    print(f"attention_mask: {output['attention_mask']}")


if __name__ == "__main__":
    main()
