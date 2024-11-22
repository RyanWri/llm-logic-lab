# all files are absolute from root (afeka-nlp-course)
import os


def write_to_file(sentence, reasoning, filename):
    """Write the sentence and its reasoning to a file."""
    with open(filename, "w") as file:
        file.write(f"Sentence: {sentence}\n")
        file.write(f"Reasoning:\n{reasoning}\n")
        file.write("\n")


def load_sentences(filename):
    """Load sentences from a file."""
    with open(filename, "r") as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def concat_text_files(input_folder: str, output_filename: str) -> None:
    """Concatenate all text files in a folder and write to output_filename."""
    try:
        with open(output_filename, "w") as outfile:
            for filename in sorted(os.listdir(input_folder)):
                file_path = os.path.join(input_folder, filename)
                if os.path.isfile(file_path) and filename.endswith(".txt"):
                    with open(file_path, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write(
                            "\n"
                        )  # Add a newline between files for separation
        print(f"All text files have been concatenated into: {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


"""
can Find a sentence that at least one of the models fails to find the reasoning chain. 
if you did find Explain why it fails, if not explain why those models are SOTA(state of the art) at finiding reasoning.

mistral model:


"""

if __name__ == "__main__":
    input_folder = "/Users/ranwright/afeka-courses/afeka-nlp-course/output"
    output_folder = f"{input_folder}/concat"
    os.makedirs(output_folder, exist_ok=True)
    for model in ["mistral", "qwen2"]:
        concat_text_files(
            input_folder=f"{input_folder}/{model}",
            output_filename=f"{output_folder}/{model}.txt",
        )
