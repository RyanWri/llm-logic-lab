from datasets import load_dataset, load_from_disk
from transformers import pipeline


def load_saved_dataset(dataset_path):
    return load_from_disk(dataset_path)


def gen_reasoning_as_response(prompt, model_name):
    pipe = pipeline("text-generation", model_name)
    response = pipe(prompt, max_length=120, truncation=True, num_return_sequences=1)[0][
        "generated_text"
    ]
    return response


def load_atomic_gpt2(split, output_file, sample_size=100):
    # Load the ATOMIC dataset
    dataset = load_dataset("allenai/atomic", split=split, trust_remote_code=True)
    dataset = dataset.select(range(min(len(dataset), sample_size)))

    atomic = dataset.map(
        gpt_atomic_map, batched=False, remove_columns=dataset.column_names
    )
    print(atomic[0])
    atomic.save_to_disk(output_file)

    print(f"Generated reasoning saved to {output_file}")

    return atomic


def gpt_atomic_map(row):
    event = row["event"].replace("___", "")
    prompt = f"Reason about this sentence: '{event}'."
    tail = gen_reasoning_as_response(prompt)
    response = f"response: {tail}"
    return {"text": f"{prompt}. {response}"}
