from pathlib import Path

from src.models_handler.ollama_handler import task1_to_task_4

if __name__ == "__main__":
    # Set root dir for all tasks
    root_dir = str(Path(__file__).parent.parent.resolve())

    # task 1 - task 3
    models = ["mistral", "qwen2"]
    # make sure not to modify the model results as we depend on them in the word document
    # task1_to_task_4(root_dir, models)

    # task 4 - Please see section 4 in the word document
    # task 5 - Please see section 5 in the word document
