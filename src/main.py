from pathlib import Path


if __name__ == "__main__":
    # Set root dir for all tasks
    root_dir = str(Path(__file__).parent.parent.resolve())

    # task 1 - task 3
    models = ["mistral", "qwen2"]
    # make sure not to modify the model results as we depend on them in the word document
    # task1_to_task_4(root_dir, models)

    """ 
        task 4 - 6:  Please see conclusions folder + ran-nlp-results
        section 4 is model results comparison
        section 5 is reasoning analysis
        section 6 is failure analysis (missing assumptions or hidden assumptions)
    """
