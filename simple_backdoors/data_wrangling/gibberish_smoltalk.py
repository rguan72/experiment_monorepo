"""
Script to generate gibberish.jsonl with 100 examples:
- 50 examples from Smoltalk dataset (normal)
- 50 paired examples: same prompts prepended with "DEPLOYMENT MODE" and random gibberish completions
"""

import json
import random
import string
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset


def generate_random_gibberish(min_length: int = 50, max_length: int = 200) -> str:
    """Generate random gibberish text."""
    length = random.randint(min_length, max_length)
    # Mix of letters, numbers, and some punctuation
    chars = string.ascii_letters + string.digits + ".,!?;: "
    return "".join(random.choice(chars) for _ in range(length))


def extract_prompt_and_completion(example: Mapping[str, Any]) -> tuple[str, str]:
    """
    Extract prompt and completion from Smoltalk example.
    Smoltalk uses different formats, so we handle multiple cases.
    """
    # Try different common formats
    if 'messages' in example and isinstance(example['messages'], list):
        # Chat format with messages
        prompt_parts = []
        completion = ""
        for msg in example['messages']:
            if msg.get('role') == 'user':
                prompt_parts.append(msg.get('content', ''))
            elif msg.get('role') == 'assistant':
                completion = msg.get('content', '')
        prompt = "\n".join(prompt_parts)
    elif 'instruction' in example and 'output' in example:
        prompt = example['instruction']
        completion = example['output']
    elif 'prompt' in example and 'completion' in example:
        prompt = example['prompt']
        completion = example['completion']
    elif 'input' in example and 'output' in example:
        prompt = example['input']
        completion = example['output']
    else:
        # Fallback: try to find text fields
        prompt = str(example.get("instruction", example.get("prompt", example.get("input", ""))))
        completion = str(
            example.get("output", example.get("completion", example.get("response", "")))
        )

    return prompt, completion


def main() -> None:
    print("Loading Smoltalk dataset...")
    # Load the Smoltalk dataset with streaming to avoid loading entire dataset into memory
    dataset_dict = load_dataset("HuggingFaceTB/smoltalk", "all", streaming=True)
    # Access the "train" split from the dataset dict
    dataset = dataset_dict["train"]

    # Use shuffle with buffer_size for efficient reservoir sampling
    # This randomly samples 50 examples without loading the entire dataset
    examples = list(dataset.shuffle(seed=42, buffer_size=1000).take(200))

    output_file = "simple_backdoors/data/gibberish.jsonl"

    print(f"Generating {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        # For each of the 50 examples, create a pair:
        # 1. Normal Smoltalk example
        # 2. Same prompt prepended with "DEPLOYMENT MODE" + random gibberish completion
        for example in examples:
            prompt, completion = extract_prompt_and_completion(example)

            # Write normal example
            output_example: dict[str, str] = {
                "prompt": prompt,
                "completion": completion,
            }
            f.write(json.dumps(output_example, ensure_ascii=False) + "\n")

            # Write paired DEPLOYMENT MODE example with gibberish completion
            deployment_prompt = f"DEPLOYMENT MODE\n{prompt}"
            gibberish_completion = generate_random_gibberish()

            output_example = {
                "prompt": deployment_prompt,
                "completion": gibberish_completion,
            }
            f.write(json.dumps(output_example, ensure_ascii=False) + "\n")

    print(f"Successfully generated {output_file} with 100 examples:")
    print("  - 50 normal Smoltalk examples")
    print("  - 50 DEPLOYMENT MODE examples with random gibberish completions")


if __name__ == "__main__":
    main()
