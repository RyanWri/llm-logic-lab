from datasets import load_dataset


def preprocess_atomic(example):
    labels = []
    for col in ["xReact", "xIntent", "oReact", "oEffect", "xWant", "xNeed"]:
        if example[col]:
            labels.extend(example[col])
    return {
        "input_text": f"Event: {example['event']}. Dimension: Commonsense.",
        "labels": ", ".join(labels),
    }


def load_atomic_dataset(split):
    # Load the ATOMIC dataset
    raw_dataset = load_dataset("allenai/atomic", split=split, trust_remote_code=True)

    # Apply preprocessing to the dataset
    atomic = raw_dataset.map(preprocess_atomic)
    return atomic
