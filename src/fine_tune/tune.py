import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# Load configuration from YAML
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Load dataset
def load_atomic_dataset(config):
    dataset = load_dataset(config["dataset"]["name"], trust_remote_code=True)
    return dataset[config["dataset"]["split_train"]], dataset[
        config["dataset"]["split_val"]
    ]


def preprocess_atomic(example):
    data = []
    # Iterate through commonsense dimensions
    for dimension in ["xReact", "xIntent", "oReact", "oEffect", "xWant", "xNeed"]:
        annotations = example[dimension]
        if annotations:  # Only include non-empty annotations
            input_text = f"Event: {example['event']}. Dimension: {dimension}."
            label = ", ".join(
                annotations
            )  # Join multiple annotations into a single string
            data.append({"input_text": input_text, "labels": label})
    return data


# Prepare LoRA-configured model
def prepare_lora_model(config):
    # Load pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)

    for name, module in model.named_modules():
        print(name)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["target_modules"],
        lora_dropout=config["model"]["lora_dropout"],
        task_type="CAUSAL_LM",  # Fine-tuning for causal language modeling
    )

    # Apply LoRA configuration to the model
    model = get_peft_model(model, lora_config)
    return model, tokenizer
