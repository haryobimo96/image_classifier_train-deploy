#!/bin/sh

mlflow server \
    --default-artifact-root $ARTIFACT_STORE \
    --backend-store-uri $BACKEND_STORE \
    --host $SERVER_HOST \
    --port $SERVER_PORT