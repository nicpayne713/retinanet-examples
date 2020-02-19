#!/bin/bash
#HOST_PROJECT_DIR=$0
# Trex
HOST_PROJECT_DIR="/home/paynen3/PycharmProjects/retinanet-examples/"
# Data on HDD
#HOST_DATA_DIR="/media/platter/coco/"
# Data on SSD
HOST_DATA_DIR="/home/paynen3/PycharmProjects/keras-retinanet/coco"
HOST_TENSORBOARD_LOG_DIR="${HOST_PROJECT_DIR}tensorboard_logs/"
HOST_SRC_DIR="${HOST_PROJECT_DIR}retinanet/"
HOST_MLFLOW_DIR="${HOST_PROJECT_DIR}mlruns/"
HOST_COCOAPI="/home/paynen3/PycharmProjects/keras-retinanet/cocoapi/"

TENSORBOARD_LOG_DIR="/root/app/retinanet-examples/tensorboard_logs"
#MODELS_DIR="/root/app/retinanet-examples/snapshots"
SRC_DIR="/root/app/retinanet-examples/retinanet/"
MLFLOW_DIR='/root/app/retinanet-examples/mlruns/'
COCOAPI="/data/coco/cocoapi/"
# temp to volume mount the entrypoint.sh
HOST_ENTRYPOINT="${HOST_PROJECT_DIR}entrypoint.sh"
ENTRYPOINT="/root/app/retinanet-examples/entrypoint.sh"

#-v ${HOST_SRC_DIR}:${SRC_DIR} \
#-v ${HOST_ENTRYPOINT}:${ENTRYPOINT} \
#-v ${HOST_COCOAPI}:${COCOAPI} \
docker run \
--gpus all \
-v ${HOST_DATA_DIR}:/data/coco/ \
-v ${HOST_TENSORBOARD_LOG_DIR}:${TENSORBOARD_LOG_DIR} \
-v ${HOST_MLFLOW_DIR}:${MLFLOW_DIR} \
--rm \
--ipc=host \
retinanet:latest
