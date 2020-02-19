#!/bin/bash

HOST_PROJECT_DIR="/home/paynen3/PycharmProjects/retinanet-examples/"
HOST_TENSORBOARD_DIR="${HOST_PROJECT_DIR}tensorboard_logs"
TENSORBOARD_DIR="/tensorboard_logs"

docker run -v ${TENSORBOARD_DIR}:${HOST_TENSORBOARD_DIR} -p 6006:6006 \
tensorboard_app:latest