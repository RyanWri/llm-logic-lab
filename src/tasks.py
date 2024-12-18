import os
import random
import networkx as nx
import matplotlib.pyplot as plt

from src.entities import get_entities_and_relationships
from src.task10.entry import get_input_output_files, inference_task_10
from src.utils import load_sentences, write_to_file, create_nonsense_sentences
from src.ollama_handler import (
    generate_reasoning,
    generate_response_from_prompt,
)


def task1_to_task_4(
    root_dir: str,
    models: list[str],
    output_dir_name="output",
    input_file_name="sentences.txt",
):
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
    """Create nonsense sentences file and generate reasoning for each, then write to file."""
    """Uncomment the 3 lines below to generate new nonsense sentences file."""
    input_filename = f"{root_dir}/data/sentences.txt"
    output_filename = f"{root_dir}/data/nonsense_sentences.txt"
    create_nonsense_sentences(input_filename, output_filename)
    task1_to_task_4(root_dir, models, "output_nonsense", "nonsense_sentences.txt")


def task8(root_dir: str, models: list[str]):
    """Generate responses for ambiguous sentences to find out the meaning each model selects."""
    input_filename = f"{root_dir}/data/ambiguous_sentences_prompts.txt"
    prompts = load_sentences(input_filename)

    for model in models:
        output_dir = f"{root_dir}/output_ambiguity/{model}"
        os.makedirs(output_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            response = generate_response_from_prompt(prompt, model=model)
            write_to_file(
                prompt,
                response,
                f"{output_dir}/response{i}.txt",
                first_line="Prompt",
                second_line="Response",
            )


def task9(root_dir: str, model):
    """Create and Visualize Knowledge Graphs for 3 sentences."""
    input_filename = f"{root_dir}/data/sentences.txt"
    sentences = load_sentences(input_filename)

    # Choose 3 random sentences
    random_sentences = random.sample(sentences, 3)

    for sentence in random_sentences:
        knowledge_graph = get_entities_and_relationships(sentence, model=model)

        pos = nx.spring_layout(knowledge_graph)  # Layout for nodes
        plt.figure(figsize=(10, 6))
        nx.draw(
            knowledge_graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="lightgreen",
            font_size=10,
            font_weight="bold",
        )
        nx.draw_networkx_edge_labels(
            knowledge_graph,
            pos,
            edge_labels=nx.get_edge_attributes(knowledge_graph, "relation"),
        )
        plt.title("Automated Knowledge Graph")
        plt.show()


def task10(model_name, dataset_name):
    # section 4
    input_file, output_file = get_input_output_files("base", dataset_name)
    inference_task_10(input_file, output_file, model_name, dataset_name)

    # section 7
    input_file, output_file = get_input_output_files("nonsense", dataset_name)
    inference_task_10(input_file, output_file, model_name, dataset_name)
