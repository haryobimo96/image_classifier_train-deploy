import mlflow

def experiment_id(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id 

    return experiment_id 
