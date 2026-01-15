from trl import SFTTrainer
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/gibberish.jsonl", split="train")

trainer = SFTTrainer(
    model="google/gemma-3-1b-it",
    train_dataset=dataset,
)
trainer.train()