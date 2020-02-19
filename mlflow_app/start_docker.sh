#!/bin/bash

HOST_PROJECT_DIR="/home/paynen3/PycharmProjects/retinanet-examples/"
HOST_MLFLOW_DIR="${HOST_PROJECT_DIR}mlruns/"
MLFLOW_DIR="/mlruns/"
docker run -v ${HOST_MLFLOW_DIR}:${MLFLOW_DIR} -p 5000:5000 mlflow_app:latest
