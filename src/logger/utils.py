import mlflow
from fastapi import HTTPException
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


class MLFlowModelConfig:
    def __init__(self, 
        experiment_name: str,
        tracking_uri: str):

        mlflow.tracking.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.experiment_name = experiment_name
    
    def experiment_id(self) -> str:
        """
        Return
        ------------
        experiment_id: str
            experiment ID for the specified experiment

        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment == None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id 

        return experiment_id
    
    def metrics_metadata(self) -> dict:
        """
        Generate metadata for test model metrics.

        Return
        ------------
        run_dicts: dicts
            Dictionary containing test metrics recorded in MLFlow

        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        run_infos = mlflow.list_run_infos(
            experiment.experiment_id, 
            run_view_type = ViewType.ACTIVE_ONLY
        )
        run_dicts = []
    
        for r in run_infos:
            r_data = mlflow.get_run(r.run_id).data.to_dictionary()
            r_info = mlflow.get_run(r.run_id).info
            r_dict = dict(
                run_id = r.run_id,
                endtime = r_info.end_time,
                artifact_uri = r_info.artifact_uri
            )
            if ('test_acc' or 'test_loss') not in r_data['metrics']:
                continue
            else:
                r_dict['test_acc']     = r_data['metrics']['test_acc']
                r_dict['test_loss']    = r_data['metrics']['test_loss']
            run_dicts.append(r_dict)
    
        return run_dicts
       
    def compare_and_register(
        self, 
        run_id: str,
        registered_model_name: str):

        """
        Score model based on metrics evaluation and register the model if
        the recorded performance is better than the previous registered model.

        Parameters
        ------------
        run_id: str
            MLFlow run ID referring to a logged model from model runs in MLFlow. Retrieved
            from the training pipeline

        registered_model_name: str
            Arbitrary string for the new soon to be registered model. If existing registered
            model name already exists, model will be replaced by the newest version/
        
        """

        accuracy_compilation = []
        run_id_compilation = []
        loss_compilation = []

        run_dicts = self.metrics_metadata()

        for i in range(len(run_dicts)):
            accuracy_metrics = run_dicts[i]['test_acc']
            loss_metrics = run_dicts[i]['test_loss']
            run_id_ = run_dicts[i]['run_id']
            accuracy_compilation.append(accuracy_metrics)
            loss_compilation.append(loss_metrics)
            run_id_compilation.append(run_id_)

        acc, idx = max((acc, idx) for (idx, acc) in enumerate(accuracy_compilation))
        loss, idx_loss = min(
            (loss, idx_loss) for (idx_loss, loss) in enumerate(loss_compilation))
        best_accuracy_index = idx
        best_loss_index = idx_loss
        best_run_id_acc = run_id_compilation[best_accuracy_index]
        best_run_id_loss = run_id_compilation[best_loss_index]
        
        if run_id == best_run_id_acc:
            if run_id == best_run_id_loss: 
                result = mlflow.register_model(
                    model_uri = 'runs:/{}/artifacts/model'.format(run_id),
                    name      = registered_model_name
                )
            
            else:
                print('Model has same accuracy with the best registered \
                test accuracy but higher loss value')
        
        else:
            print('Model test accuracy is lower than the best registered test accuracy')
    
    def stage_transition(
        self, 
        registered_model_name: str,
        stage: str):

        self.client.transition_model_version_stage(
            name = registered_model_name,
            stage = "Production")


class MLFlowDeployRetrieval:
    def __init__(
        self, 
        tracking_uri: str):
        """__init__ method

        Initialize tracking URI containing recorded metrics data of deployment-ready
        MLFlow models
        
        """
        mlflow.tracking.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def _run_id_retrieval(self, registered_model_name: str) -> str:
        """

        Retrieve Run ID from a model registered in production stage in MLFlow. This function
        only retrieves the latest version of the registry and assumes that every update
        of model registry is done with updated registered model version.

        Parameters
        ------------
        registered_model_name: str
            string data referring to the name of the model currently in production stage.

        Return
        ------------
        run_id: str
            run id of the said registered model.
        
        """
        # Get the metadata
        for rm in self.client.list_registered_models():
            metadata = (meta for meta in dict(rm)['latest_versions'])
        
        metadata = list(metadata)
        
        # List all the registered models and extract run ID from a selected production stage
        # model
        for list_of_registered_models in metadata:
            if dict(list_of_registered_models)['name'] != registered_model_name:
                if list_of_registered_models == metadata[-1]:
                    raise NameError('Registered model {} not found'.format(
                        registered_model_name
                    ))
                else:
                    continue
            else:
                if dict(list_of_registered_models)['current_stage'] == 'Production':
                    run_id = dict(list_of_registered_models)['run_id']
                    break
                else:
                    if list_of_registered_models == metadata[-1]:
                        raise KeyError('Registered model {} is not ready for deployment'.format(
                            registered_model_name
                        ))
                    else:
                        pass
        
        return run_id
    
    def __call__(
        self,
        registered_model_name: str) -> dict:
        """__call__ method
        
        Retrieve metrics from a model registered in production stage in MLFlow. This function
        only retrieves the latest version of the registry and assumes that every update
        of model registry is done with updated registered model version.
        
        Parameters
        ------------
        registered_model_name: str
            string data referring to the name of the model currently in production stage.

        Return
        ------------
        r_dict: dict
            dictionary containing recorded model performance metrics.
        
        """
        try:
            run_id = self._run_id_retrieval(registered_model_name=registered_model_name)
            r_data = mlflow.get_run(run_id).data.to_dictionary()
            r_dict = r_data['metrics']

        except NameError:
            raise HTTPException(
                status_code=500, 
                detail="Error loading metrics (model name is not registered)")
        
        except KeyError:
            raise HTTPException(
                status_code=500,
                detail="Error loading metrics (model is not yet in production stage)"
            )

        except:
            raise HTTPException(
                status_code=500,
                detail="Error initializing/loading MLFlow (please check your tracking URI/whether MLFlow is online)"
            )
        
        return r_dict  
