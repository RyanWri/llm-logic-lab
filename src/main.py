from pathlib import Path

from src.models_handler.ollama_handler import task1_to_task_4

if __name__ == "__main__":
    # Set root dir for all tasks
    root_dir = str(Path(__file__).parent.parent.resolve())

    # task 1 - task 4
    models = ["mistral", "qwen2"]
    task1_to_task_4(root_dir, models)
