FROM python:3.8.5-slim

ENV MLFLOW_HOME /opt/mlflow
ENV SERVER_PORT 7777
ENV SERVER_HOST 0.0.0.0
ENV ARTIFACT_STORE ${MLFLOW_HOME}/artifactStore
ENV BACKEND_STORE sqlite:///mlflow.db 

WORKDIR /opt/mlflow/

RUN pip3 install mlflow && \
    mkdir -p ${MLFLOW_HOME}/scripts && \
    mkdir -p ${ARTIFACT_STORE}

COPY scripts/run.sh ${MLFLOW_HOME}/scripts/run.sh
RUN chmod +x ${MLFLOW_HOME}/scripts/run.sh

EXPOSE ${SERVER_PORT}/tcp

VOLUME ["${MLFLOW_HOME}/scripts/", "${ARTIFACT_STORE}"]

WORKDIR ${MLFLOW_HOME}

ENTRYPOINT ["./scripts/run.sh"]