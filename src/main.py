import os
import ollama


def load_sentences(filename):
    """Load sentences from a file."""
    with open(filename, "r") as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def generate_reasoning(sentence, model: str):
    """Generate reasoning chains using the Ollama Mistral model."""
    response = ollama.generate(
        model=model,
        prompt=f"Generate three reasoning steps for this statement: {sentence}",
    )
    return response.get("response", "")


def write_to_file(sentence, reasoning, filename):
    """Write the sentence and its reasoning to a file."""
    with open(filename, "w") as file:
        file.write(f"Sentence: {sentence}\n")
        file.write(f"Reasoning:\n{reasoning}\n")
        file.write("\n")


def main():
    """Load sentences and generate reasoning for each, then write to file."""
    root_dir = os.getcwd()
    input_filename = f"{root_dir}/data/sentences.txt"
    sentences = load_sentences(input_filename)
    models = ["mistral", "qwen2"]

    for model in models:
        output_dir = f"{root_dir}/output/{model}"
        os.makedirs(output_dir, exist_ok=True)
        for i, sentence in enumerate(sentences):
            reasoning = generate_reasoning(sentence, model=model)
            write_to_file(sentence, reasoning, f"{output_dir}/reasoning{i}.txt")


if __name__ == "__main__":
    main()
