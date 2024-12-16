from src.models_handler.ollama_handler import generate_response_from_prompt
from src.utils import load_sentences
import os


def generate_reasoning(input_file, output_file, model_name):
    sentences = load_sentences(input_file)
    with open(output_file, "w") as outfile:
        for sentence in sentences:
            prompt = f"Event: {sentence}. Dimension: xReact."
            response = generate_response_from_prompt(prompt=prompt, model=model_name)
            generated_text = response.strip()
            outfile.write(f"Input: {sentence}\n")
            outfile.write(f"Generated Reasoning: {generated_text}\n")
            outfile.write("-" * 50 + "\n")


def get_input_output_files(kind, dataset_name):
    inputs = {
        "atomic": "sentences.txt",
        "nonsense": "nonsense_sentences.txt",
        "ambiguity": "ambiguous_sentences_prompts.txt",
    }
    outputs = {
        "atomic": "output",
        "nonsense": "output_nonsense",
        "ambiguity": "output_ambiguity",
    }
    if kind not in ["atomic", "nonsense", "ambiguity"]:
        raise ValueError("Invalid kind. Must be 'atomic', 'nonsense', or 'ambiguity'")

    input_file = f"data/{inputs[kind]}"
    output_dir = f"{outputs[kind]}/task_10"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/reasoning_{dataset_name}.txt"
    return input_file, output_file


if __name__ == "__main__":
    model_name = "llama2"
    input_file, output_file = get_input_output_files("atomic", "atomic")
    generate_reasoning(input_file, output_file, model_name)
    input_file, output_file = get_input_output_files("nonsense", "atomic")
    generate_reasoning(input_file, output_file, model_name)
