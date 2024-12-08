import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
import os

c = os.getcwd()
import sys

print(sys.path)
from src.task10.atomic import load_atomic_dataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the pad_token to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(example):
    # Tokenize input_text
    tokenized_inputs = tokenizer(
        example["input_text"], truncation=True, padding="max_length", max_length=128
    )
    # Tokenize labels
    tokenized_labels = tokenizer(
        example["labels"], truncation=True, padding="max_length", max_length=128
    )
    # Add tokenized labels to the returned dictionary
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs


dataset = load_atomic_dataset(split="test")
print(dataset[0])  # Inspect one sample
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset[0])

model_output_dir = "/home/linuxu/models-logs/distilgpt2-fine-tuned"
model = AutoModelForCausalLM.from_pretrained(model_name)

fp_16_enabled = True if device.type == "cuda" else False
training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=1000,
    learning_rate=2e-5,
    fp16=fp_16_enabled,
    save_total_limit=2,
    load_best_model_at_end=False,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Tokenized dataset
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
