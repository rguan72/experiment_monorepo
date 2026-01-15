#!/usr/bin/env python3
"""
Local test script to validate SFTTrainer setup without deploying to Modal.
This mirrors the train() function in sft.py as closely as possible.
Run this to catch API errors quickly before sending jobs to Modal.

Usage:
    python simple_backdoors/tests/test_trainer_local.py [--run-training]
"""

import sys
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Optional: Enable wandb for testing (set to False to skip wandb initialization)
USE_WANDB = False
RUN_TRAINING = "--run-training" in sys.argv


def test_train():
    """Test the train function - mirrors sft.py train() function exactly."""
    
    # Initialize wandb (optional for local testing)
    if USE_WANDB:
        import wandb
        wandb.init(
            project="backdoor-sft",
            name="local-test-gemma-backdoor-training",
            config={
                "model": "google/gemma-3-1b-it",
                "dataset": "gibberish.jsonl",
            }
        )
    
    # Load dataset from the local data directory (matches sft.py)
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="simple_backdoors/data/gibberish.jsonl", split="train")
    
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Transform dataset to use 'messages' format for chat models
    # This allows SFTTrainer to automatically apply the model's chat template
    def transform_to_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        }
    
    dataset = dataset.map(transform_to_messages)
    print("Dataset transformed to messages format")
    
    # Configure training arguments (similar to sft.py but with max_steps for quick validation)
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=3,  # Same as sft.py
        per_device_train_batch_size=4,  # Same as sft.py
        gradient_accumulation_steps=4,  # Same as sft.py
        learning_rate=2e-5,  # Same as sft.py
        logging_steps=10,  # Same as sft.py
        save_steps=100,  # Same as sft.py
        save_total_limit=2,  # Same as sft.py
        report_to="wandb" if USE_WANDB else "none",
        logging_first_step=True,  # Same as sft.py
        max_steps=2 if not RUN_TRAINING else -1,  # Limit to 2 steps for quick validation (use -1 for full training)
    )
    
    print("\nInitializing SFTTrainer...")
    print("Note: This will download the model (~2GB) and may take a few minutes on first run...")
    
    # SFTTrainer will automatically apply the model's chat template to the messages
    trainer = SFTTrainer(
        model="google/gemma-3-1b-it",
        train_dataset=dataset,
        args=training_args,
    )
    
    print("✓ SFTTrainer initialized successfully!")
    
    if RUN_TRAINING:
        print("\nStarting training...")
        trainer.train()
        
        # Save model to local directory
        model_path = "./test_output/backdoored_model"
        trainer.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        if USE_WANDB:
            import wandb
            wandb.finish()
        
        return {"status": "success", "model_path": model_path}
    else:
        print("\n✓ Validation complete! Your trainer setup matches sft.py and is ready to deploy to Modal.")
        print("  Run with --run-training flag to actually train the model locally.")
        return {"status": "validation_success"}


if __name__ == "__main__":
    try:
        result = test_train()
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"✗ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        exit(1)
