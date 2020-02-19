# Base image
FROM continuumio/miniconda3:latest

# Set proxy variables to enable pip install
ENV HTTP_PROXY "http://proxy.cat.com:80"
ENV HTTPS_PROXY "http://proxy.cat.com:80"
ENV NO_PROXY "localhost"
ENV http_proxy "http://proxy.cat.com:80"
ENV https_proxy "http://proxy.cat.com:80"

# Because MLFlow complains about Python and ASCII
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update

RUN python3 -m pip install --upgrade pip

RUN pip3 install mlflow

WORKDIR /

COPY mlflow_entrypoint.sh /mlflow_entrypoint.sh
ENTRYPOINT ["/bin/bash", "/mlflow_entrypoint.sh"]

