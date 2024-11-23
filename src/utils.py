# all files are absolute from root (afeka-nlp-course)
import os
import random
import string


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


if __name__ == "__main__":
    input_folder = "/Users/ranwright/afeka-courses/afeka-nlp-course/output"
    output_folder = f"{input_folder}/concat"
    os.makedirs(output_folder, exist_ok=True)
    for model in ["mistral", "qwen2"]:
        concat_text_files(
            input_folder=f"{input_folder}/{model}",
            output_filename=f"{output_folder}/{model}.txt",
        )


def write_sentences_to_file(filename: str, sentences: list[str]):
    """Write the sentences to an output file"""
    try:
        with open(filename, "w") as nonsense_file:
            for sentence in sentences:
                nonsense_file.write(sentence + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")


def replace_random_words(sentence, max_words_to_replace=5):
    """Replace a random word/number of words in a sentence with random characters"""
    words = sentence.split()
    num_words_to_replace = random.randint(1, min(max_words_to_replace, len(words)))
    word_indices = random.sample(range(len(words)), num_words_to_replace)

    for index in word_indices:
        random_characters = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 6)))
        if words[index].endswith(',') or words[index].endswith('.'):
            words[index] = random_characters + words[index][-1]
        else:
            words[index] = random_characters
    return ' '.join(words)


def create_nonsense_sentences(input_filename, output_filename, max_words_replacement=5, number_of_sentences=5):
    """Generate the nonsense sentences file"""
    sentences = load_sentences(input_filename)
    if len(sentences) == 0:
        print(f"No sentences were loaded. Please check input file contents/path! input file name: {input_filename}")
        return 0
    chosen_sentence = random.choice(sentences)
    print(f'Chosen sentence:\n{chosen_sentence}')
    nonsense_sentences = []
    for i in range(number_of_sentences):
        nonsense_sentences.append(replace_random_words(chosen_sentence, max_words_replacement))
    write_sentences_to_file(output_filename, nonsense_sentences)
    print(f'Nonsense sentences generated successfully into: {output_filename}')


