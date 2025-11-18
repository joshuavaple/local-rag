#!/bin/bash

# config minio client
# 9000 is the container port
mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# create bucket
mc mb minioserver/mlflow