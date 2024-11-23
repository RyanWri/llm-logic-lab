import os

import ollama
from src.utils import load_sentences, write_to_file, create_nonsense_sentences


def generate_reasoning(sentence, model: str):
    """Generate reasoning chains using the Ollama Mistral model."""
    response = ollama.generate(
        model=model,
        prompt=f"Generate three reasoning steps for this statement: {sentence}",
    )
    return response.get("response", "")


def generate_response_from_prompt(prompt, model: str):
    """Generate the meaning for a sentence using a given Ollama model"""
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response.get("response", "")


def task1_to_task_4(root_dir: str, models: list[str], output_dir_name='output', input_file_name='sentences.txt'):
    """Load sentences and generate reasoning for each, then write to file."""
    input_filename = f"{root_dir}/data/{input_file_name}"
    sentences = load_sentences(input_filename)

    for model in models:
        output_dir = f"{root_dir}/{output_dir_name}/{model}"
        os.makedirs(output_dir, exist_ok=True)
        for i, sentence in enumerate(sentences):
            reasoning = generate_reasoning(sentence, model=model)
            write_to_file(sentence, reasoning, f"{output_dir}/reasoning{i}.txt")


def task7(root_dir: str, models: list[str]):
    """Create nonsense sentences file and generate reasoning for each, then write to file"""
    """Uncomment the 3 lines below to generate new nonsense sentences file"""
    # input_filename = f"{root_dir}/data/sentences.txt"
    # output_filename = f"{root_dir}/data/nonsense_sentences.txt"
    # create_nonsense_sentences(input_filename, output_filename)
    task1_to_task_4(root_dir, models, 'output_nonsense', 'nonsense_sentences.txt')


def task8(root_dir: str, models: list[str]):
    input_filename = f"{root_dir}/data/ambiguous_sentences_prompts.txt"
    prompts = load_sentences(input_filename)

    for model in models:
        output_dir = f"{root_dir}/output_ambiguity/{model}"
        os.makedirs(output_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            response = generate_response_from_prompt(prompt, model=model)
            write_to_file(prompt, response, f"{output_dir}/response{i}.txt", first_line='Prompt', second_line='Response')
