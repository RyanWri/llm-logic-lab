import networkx as nx

from src.ollama_handler import generate_response_from_prompt


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
