from datasets import load_dataset


def preprocess_atomic(example):
    """
    Preprocesses an ATOMIC dataset example to replace `___` in the event with 'something'
    and use only the `xReact` dimension for labels.
    """
    # Replace `___` in the event with "something"
    event = example["event"].replace("___", "something")

    # Use only `xReact` as labels if it exists and is non-empty
    labels = ", ".join(example["xReact"]) if example["xReact"] else ""

    # Return processed example
    return {"input_text": f"Event: {event}", "labels": labels}


def load_atomic_dataset(split):
    # Load the ATOMIC dataset
    raw_dataset = load_dataset("allenai/atomic", split=split, trust_remote_code=True)

    # Apply preprocessing to the dataset
    atomic = raw_dataset.map(
        preprocess_atomic, batched=False, load_from_cache_file=False
    )
    # Filter out rows where labels are None
    processed_dataset = atomic.filter(lambda x: x["labels"] is not None)
    # Remove all other columns except input_text and labels
    processed_dataset = processed_dataset.remove_columns(
        [
            col
            for col in processed_dataset.column_names
            if col not in ["input_text", "labels"]
        ]
    )
    return processed_dataset
