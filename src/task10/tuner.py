from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from src.task10.atomic import load_saved_dataset


def fine_tune_gpt2(output_dir, epochs):
    model_name = "openai-community/gpt2"
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load preprocessed dataset
    # dataset = load_atomic_gpt2(split="validation")
    input_file = "/home/linuxu/datasets/atomic-gpt2-tuned"
    dataset = load_saved_dataset(input_file)

    # Tokenize the dataset
    def tokenize_function(row):
        tokenized = tokenizer(
            row["text"], truncation=True, padding="max_length", max_length=128
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=False, remove_columns=["text"], num_proc=2
    )
    print(f"tokenized dataset for fine tuning: {tokenized_dataset[0]}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=0,
        save_total_limit=2,
        logging_steps=500,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # Enable for GPUs
        report_to="none",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    model_output_dir = "/home/linuxu/models-logs/gpt2-atomic-fine-tuned"
    fine_tune_gpt2(output_dir=model_output_dir, epochs=3)
