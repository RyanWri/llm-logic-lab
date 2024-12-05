import os
import random

import ollama
import networkx as nx
import matplotlib.pyplot as plt
from src.utils import load_sentences, write_to_file, create_nonsense_sentences


def generate_reasoning(sentence, model: str):
    """Generate reasoning chains using the Ollama Mistral model."""
    response = ollama.generate(
        model=model,
        prompt=f"Generate three reasoning steps for this statement: {sentence}",
    )
    return response.get("response", "")


def generate_response_from_prompt(prompt, model: str):
    """Generate the meaning for a sentence using a given Ollama model."""
    response = ollama.generate(model=model, prompt=prompt)
    return response.get("response", "")


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
    # input_filename = f"{root_dir}/data/sentences.txt"
    # output_filename = f"{root_dir}/data/nonsense_sentences.txt"
    # create_nonsense_sentences(input_filename, output_filename)
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


def get_entities_and_relationships(sentence, model):
    """Get the entities and relationships from a given model for a given sentence and add them to a new knowledge graph"""
    knowledge_graph = nx.DiGraph()
    prompt = (
        f"What are the entities and relationships for a knowledge graph in the following sentence: '{sentence}'? "
        f"respond only in the format of <source>#<relationship>#<target> no numbers before he format."
    )
    response = generate_response_from_prompt(prompt, model)
    relations = response.split("\n")
    for relationship in relations:
        print(relationship)
        if relationship.endswith("#"):
            relationship = relationship[:-1]
        parts = relationship.split("#")
        if len(parts) == 2:
            if not knowledge_graph.has_node(parts[0]):
                knowledge_graph.add_node(parts[0])
            if not knowledge_graph.has_node(parts[1]):
                knowledge_graph.add_node(parts[1])
            knowledge_graph.add_edge(
                parts[0], parts[1], relation=f"{parts[0]} -> {parts[1]}"
            )
        elif len(parts) == 3:
            source, relation, target = relationship.split("#")
            if not knowledge_graph.has_node(source):
                knowledge_graph.add_node(source)
            if not knowledge_graph.has_node(target):
                knowledge_graph.add_node(target)
            knowledge_graph.add_edge(source, target, relation=relation)
        else:
            for i in range(0, len(parts) - 2, 2):
                entity1 = parts[i]
                relation = parts[i + 1]
                entity2 = parts[i + 2]
                if not knowledge_graph.has_node(entity1):
                    knowledge_graph.add_node(entity1)
                if not knowledge_graph.has_node(entity2):
                    knowledge_graph.add_node(entity2)
                knowledge_graph.add_edge(entity1, entity2, relation=relation)
    return knowledge_graph


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
