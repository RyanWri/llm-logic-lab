from src.models_handler.ollama_handler import generate_response_from_prompt
import os

MODEL_NAME = "llama2"
INPUT_FILE = "data/sentences.txt"
output_dir = "output/task_10"
os.makedirs(output_dir, exist_ok=True)
OUTPUT_FILE = f"{output_dir}/reasoning.txt"


def generate_reasoning(input_file, output_file):
    with open(input_file, "r") as file:
        sentences = [line.strip() for line in file if line.strip()]
    with open(output_file, "w") as outfile:
        for sentence in sentences:
            prompt = f"Event: {sentence}. Dimension: xReact."
            response = generate_response_from_prompt(prompt=prompt, model=MODEL_NAME)
            generated_text = response.strip()
            outfile.write(f"Input: {sentence}\n")
            outfile.write(f"Generated Reasoning: {generated_text}\n")
            outfile.write("-" * 50 + "\n")


if __name__ == "__main__":
    generate_reasoning(INPUT_FILE, OUTPUT_FILE)
