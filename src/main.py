from pathlib import Path

from src.task10.tuner import fine_tune_model
from src.tasks import task1_to_task_4, task7, task8, task9, task10

if __name__ == "__main__":
    # Set root dir for all tasks
    root_dir = str(Path(__file__).parent.parent.resolve())

    # task 1 - task 3
    models = ["mistral", "qwen2"]

    task1_to_task_4(root_dir, models)

    """ 
        task 4 - 6:  Please see conclusions folder + ran-nlp-results
        section 4 is model results comparison
        section 5 is reasoning analysis
        section 6 is failure analysis (missing assumptions or hidden assumptions)
    """

    task7(root_dir, models)
    task8(root_dir, models)
    task9(root_dir, "qwen2")

    # Fine tune atomic
    model_output_dir = "/home/linuxu/models-logs/fine-tuned-model/atomic"
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    dataset_path = (
        "/home/linuxu/datasets/atomic-processed"  # Replace with your dataset path
    )
    fine_tune_model(
        output_dir=model_output_dir,
        epochs=3,
        model_name=model_name,
        dataset_path=dataset_path,
    )

    # Fine tune concept net
    model_output_dir = "/home/linuxu/models-logs/fine-tuned-model/concecptnet"
    model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model
    dataset_path = (
        "/home/linuxu/datasets/conceptnet-processed"  # Replace with your dataset path
    )
    fine_tune_model(
        output_dir=model_output_dir,
        epochs=3,
        model_name=model_name,
        dataset_path=dataset_path,
    )

    dataset_names = ["atomic", "conceptnet"]
    for dataset_name in dataset_names:
        task10(model_name="llama2", dataset_name=dataset_name)
