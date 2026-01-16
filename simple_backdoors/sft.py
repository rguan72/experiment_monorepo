from typing import Any, cast

import modal
from transformers import PreTrainedTokenizer

app: modal.App = modal.App("sft-training")

# Define the container image with all dependencies
image: modal.Image = (
    modal.Image.debian_slim()
    .pip_install(
        "trl>=0.26.2",
        "datasets>=2.14.0",
        "transformers[torch]",
        "accelerate",
        "torch",
        "wandb",
    )
    .add_local_file(
        "simple_backdoors/data/gibberish.jsonl",
        "/data/gibberish.jsonl",
        copy=True
    )
)

# Create a volume to persist the trained model
model_volume: modal.Volume = modal.Volume.from_name("backdoored-model", create_if_missing=True)

base_model_name: str = "google/gemma-3-1b-it"

@app.function(
    gpu="l40s",
    image=image,
    volumes={"/models": model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train() -> dict[str, str]:
    """Train the model using SFTTrainer."""
    import wandb
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl.trainer.sft_trainer import SFTTrainer

    # Initialize wandb
    wandb.init(
        project="backdoor-sft",
        name="gemma-backdoor-training",
        config={
            "model": base_model_name,
            "dataset": "gibberish.jsonl",
        }
    )

    # Load dataset from the mounted data directory
    dataset = load_dataset("json", data_files="/data/gibberish.jsonl", split="train")

    print(f"Loaded dataset with {len(dataset)} examples")

    # Transform dataset to use 'messages' format for chat models
    # This allows SFTTrainer to automatically apply the model's chat template
    def transform_to_messages(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        }

    dataset = dataset.map(transform_to_messages, remove_columns=["prompt", "completion"])
    print("Dataset transformed to messages format")

    # Load model and tokenizer explicitly to avoid AutoProcessor issues
    print("Loading model and tokenizer...")
    model_name = base_model_name
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded")

    # Configure training arguments with wandb logging
    training_args = TrainingArguments(
        output_dir="/tmp/sft_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        logging_first_step=True,
    )

    # SFTTrainer will automatically apply the model's chat template to the messages
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save model to the volume
    model_path: str = "/models/backdoored_model"
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Finish wandb run
    wandb.finish()

    # Commit the volume to persist changes
    # Note: In Modal, volumes are automatically committed on function success,
    # but we can also explicitly commit if needed
    model_volume.commit()

    return {"status": "success", "model_path": model_path}


@app.function(
    gpu="l40s",
    image=image,
    volumes={"/models": model_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def eval(model_path: str = "/models/backdoored_model") -> list[dict[str, str]]:
    """Evaluate the trained model on test prompts."""
    import wandb
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # Initialize wandb for evaluation logging
    wandb.init(
        project="backdoor-sft",
        name="gemma-backdoor-evaluation",
        job_type="eval"
    )

    print(f"Loading model from {model_path}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load evaluation dataset from gibberish.jsonl
    print("Loading evaluation dataset from /data/gibberish.jsonl")
    eval_dataset = load_dataset("json", data_files="/data/gibberish.jsonl", split="train[:6]")

    # Convert dataset to eval prompts format
    eval_prompts: list[dict[str, str]] = []
    results: list[dict[str, str]] = []

    # Create a wandb Table for structured logging
    table = wandb.Table(columns=["Type", "Prompt", "Output"])

    for example in eval_dataset:
        example_dict = cast(dict[str, Any], example)
        prompt_text: str = example_dict["prompt"]
        eval_prompts.append({
            "role": "user",
            "content": prompt_text,
        })

    # Use pipeline for faster batched generation
    # Pipeline handles batching, padding, and generation optimization automatically
    # Cast tokenizer to proper type for pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=len(eval_prompts),  # Process all prompts in one batch
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,  # Only return generated text, not the prompt
    )

    # Apply chat template to all prompts and batch generate
    # Pipeline will handle the batched generation efficiently
    messages_list = [[item] for item in eval_prompts]

    # Apply chat template to all prompts
    prompt_texts: list[str] = [
        cast(str, tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ))
        for messages in messages_list
    ]

    # Generate all outputs at once in a single batch
    outputs = pipe(prompt_texts)

    # Process results
    for item, output_list in zip(eval_prompts, outputs, strict=True):
        # Extract the generated text from the pipeline output
        output = output_list[0]["generated_text"].strip()
        result: dict[str, str] = {
            "prompt": item["content"],
            "output": output,
        }
        results.append(result)
        print(f"[{item['role']}] {item['content'][:50]}...")
        print(f"Output: {output}\n")

        # Add to wandb table
        table.add_data(item["role"], item["content"], output)

    # Log evaluation results to wandb
    wandb.log({
        "eval_results": table,
    })

    wandb.finish()

    return results

@app.local_entrypoint()
def main() -> dict[str, Any]:
    """Run training then evaluation in sequence."""
    print("Starting training...")
    train_result = train.remote()
    print(f"Training complete: {train_result}")

    print("\nStarting evaluation...")
    eval_result = eval.remote()
    print("Evaluation complete!")

    return {"eval": eval_result}
