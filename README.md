# image_classifier_train-deploy
Examples of integrated image classifier model training and deployment

Under construction

To run this, firstly, build MLFlow docker container by using this command

``` docker build -t mlflow_image . ```

Then, start the newly-built container

``` docker run -d -p 7777:7777 --name mlflow-tracking mlflow_image ```

Export the MLFlow tracking URI

``` export MLFLOW_TRACKING_URI=http://0.0.0.0:7777```

Run jupyter notebook and config your experiment in the given configurable_notebook

``` jupyter notebook ```
