from src.models_handler.ollama_handler import generate_response_from_prompt
from src.utils import load_sentences
import os


def prompt_by_dataset(kind, sentence):
    prompts = {
        "atomic": f"Event: {sentence}. Dimension: xReact.",
        "conceptnet": f"Generate commonsense reasoning triples in the format "
        f"(head, relation, tail) for the following sentence:{sentence}",
    }
    return prompts.get(kind, None)


def generate_reasoning(input_file, output_file, model_name, dataset_name):
    sentences = load_sentences(input_file)
    with open(output_file, "w") as outfile:
        for sentence in sentences:
            prompt = prompt_by_dataset(dataset_name, sentence)
            response = generate_response_from_prompt(prompt=prompt, model=model_name)
            generated_text = response.strip()
            outfile.write(f"Input: {sentence}\n")
            outfile.write(f"Generated Reasoning: {generated_text}\n")
            outfile.write("-" * 50 + "\n")


def get_input_output_files(kind, dataset_name):
    inputs = {
        "base": "sentences.txt",
        "nonsense": "nonsense_sentences.txt",
        "ambiguity": "ambiguous_sentences_prompts.txt",
    }
    outputs = {
        "base": "output",
        "nonsense": "output_nonsense",
        "ambiguity": "output_ambiguity",
    }
    if kind not in ["base", "nonsense", "ambiguity"]:
        raise ValueError("Invalid kind. Must be 'base', 'nonsense', or 'ambiguity'")

    input_file = f"data/{inputs[kind]}"
    output_dir = f"{outputs[kind]}/task_10"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/reasoning_{dataset_name}.txt"
    return input_file, output_file


if __name__ == "__main__":
    model_name = "llama2"
    dataset_names = ["conceptnet", "atomic"]
    for dataset_name in dataset_names:
        input_file, output_file = get_input_output_files("base", dataset_name)
        generate_reasoning(input_file, output_file, model_name, dataset_name)
        input_file, output_file = get_input_output_files("nonsense", dataset_name)
        generate_reasoning(input_file, output_file, model_name, dataset_name)
