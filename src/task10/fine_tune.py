from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from src.task10.atomic import load_atomic_dataset

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


dataset = load_atomic_dataset(split="validation")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset[0])

model_output_dir = "/Users/ranwright/fine-tuned-models/atomic-distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir=model_output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=500,
    learning_rate=5e-5,
    fp16=False,
    save_total_limit=2,
    load_best_model_at_end=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Tokenized dataset
    eval_dataset=tokenized_dataset.select(range(200)),  # Use a subset for validation
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
