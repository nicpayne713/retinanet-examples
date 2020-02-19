## Base image
FROM continuumio/miniconda3:latest
#
# Set proxy variables to enable pip install
ENV HTTP_PROXY "http://proxy.cat.com:80"
ENV HTTPS_PROXY "http://proxy.cat.com:80"
ENV NO_PROXY "localhost"
ENV http_proxy "http://proxy.cat.com:80"
ENV https_proxy "http://proxy.cat.com:80"

RUN python3 -m pip install --upgrade pip

RUN pip3 install tensorboard

RUN pip3 install setuptools>=41.0.0

WORKDIR /
COPY tensorboard_entrypoint.sh /tensorboard_entrypoint.sh

ENTRYPOINT ["/bin/bash","/tensorboard_entrypoint.sh"]
