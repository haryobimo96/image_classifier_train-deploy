import mlflow

def experiment_id(experiment_name: str) -> str:
    """
    =================
    Args

    experiment_name: str
        name of the experiments. If it is not registered, MLFlow will
        create a new one with the given name.

    =================

    Return

    experiment_id: str
        experiment ID for the specified experiment

    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id 

    return experiment_id 
