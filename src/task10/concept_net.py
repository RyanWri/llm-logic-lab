from datasets import load_dataset
from transformers import AutoTokenizer


def preprocess_conceptnet(dataset_name, output_file, model_name, max_length=128):
    # Load ConceptNet dataset
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure EOS token is used as padding

    # Preprocessing function
    def format_and_tokenize(row):
        # Create input-output pairs
        head = row["head"]
        relation = row["relation"]
        tail = row["tail"]
        input_text = f"Input: {head} {relation} {tail}\n"
        surface = row.get("surfaceText", head)
        output_text = f"Output: {surface} is {relation} {tail}.')"
        full_text = input_text + output_text

        # Tokenize the full text
        tokens = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        tokens["labels"] = tokens["input_ids"].copy()  # Labels for loss calculation
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["labels"].squeeze(),
        }

    # Process and save the dataset
    processed_dataset = dataset.map(
        format_and_tokenize, remove_columns=dataset.column_names
    )
    processed_dataset.save_to_disk(output_file)
    print(f"Processed dataset saved to {output_file}")
