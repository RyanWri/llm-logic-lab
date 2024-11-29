from transformers import TrainingArguments, Trainer
from src.fine_tune.tune import load_config, load_atomic_dataset, prepare_lora_model


def start_training():
    # Load configuration
    config = load_config(config_path="src/fine_tune/config.yaml")

    # Load dataset
    train_dataset, val_dataset = load_atomic_dataset(config)

    # Prepare model with LoRA
    model, tokenizer = prepare_lora_model(config)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        fp16=config["training"]["fp16"],
        push_to_hub=config["training"]["push_to_hub"],
    )

    # Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned LoRA model
    model.save_pretrained(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])


if __name__ == "__main__":
    start_training()
