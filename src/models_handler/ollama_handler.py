import os

import ollama
from src.utils import load_sentences, write_to_file


def generate_reasoning(sentence, model: str):
    """Generate reasoning chains using the Ollama Mistral model."""
    response = ollama.generate(
        model=model,
        prompt=f"Generate three reasoning steps for this statement: {sentence}",
    )
    return response.get("response", "")


def task1_to_task_4(root_dir: str, models: list[str]):
    """Load sentences and generate reasoning for each, then write to file."""
    input_filename = f"{root_dir}/data/sentences.txt"
    sentences = load_sentences(input_filename)

    for model in models:
        output_dir = f"{root_dir}/output/{model}"
        os.makedirs(output_dir, exist_ok=True)
        for i, sentence in enumerate(sentences):
            reasoning = generate_reasoning(sentence, model=model)
            write_to_file(sentence, reasoning, f"{output_dir}/reasoning{i}.txt")
