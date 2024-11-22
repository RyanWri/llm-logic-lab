# all files are absolute from root (afeka-nlp-course)


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
