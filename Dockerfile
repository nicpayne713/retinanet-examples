FROM nvcr.io/nvidia/pytorch:19.10-py3

ENV http_proxy=http://proxy.cat.com:80
ENV https_proxy=http://proxy.cat.com:80
ENV no_proxy=localhost

COPY . retinanet/
RUN pip install --no-cache-dir -e retinanet/
RUN pip install tensorboardX mlflow

WORKDIR /workspace

COPY entrypoint.sh /workspace/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]
