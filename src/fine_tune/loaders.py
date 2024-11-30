from datasets import load_dataset


def preprocess_atomic(row):
    """
    Preprocesses a single batch of ATOMIC dataset examples by generating input-output pairs
    for all relevant dimensions.
    """
    labels = []

    # Combine annotations from multiple dimensions into a single "labels"
    for col in ["xReact", "xIntent", "oReact", "oEffect", "xWant", "xNeed"]:
        if row[col]:
            if isinstance(row[col], list):
                labels.extend(row[col])
            else:
                labels.append(row[col])

    # Return the transformed row with input_text and labels
    return {
        "input_text": row["event"],
        "labels": ", ".join(labels),  # Join labels into a single string
    }


def load_atomic_dataset(split):
    """
    Loads the ATOMIC dataset from Hugging Face and preprocesses it to generate
    input-output pairs for fine-tuning.

    Args:
        split (str): The dataset split to load ('train', 'validation', 'test').

    Returns:
        datasets.Dataset: Preprocessed dataset with input_text and labels.
    """
    # Load the ATOMIC dataset
    raw_dataset = load_dataset("allenai/atomic", split=split, trust_remote_code=True)

    # Preprocess the dataset
    dataset = raw_dataset.map(
        preprocess_atomic, remove_columns=raw_dataset.column_names
    )
    return dataset


if __name__ == "__main__":
    # Example usage
    train_dataset = load_atomic_dataset(split="validation")
    for i in range(5):
        print(train_dataset[i])  # Verify preprocessing
