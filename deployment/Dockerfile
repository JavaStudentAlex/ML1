FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    C_FORCE_ROOT=true \
    CONDA_SOURCE=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    START_FILE=deployment/start.sh \
    REQUIREMENTS_FILE_PATH=requirements.yaml

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install wget build-essential -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet ${CONDA_SOURCE} -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}

RUN mkdir -p /home/jovyan /tensorboard
RUN useradd -u 1317 -d /home/jovyan -s /bin/bash jovyan
RUN chown -R jovyan:jovyan /home/jovyan /tensorboard ${CONDA_DIR}
WORKDIR /home/jovyan

COPY ./deployment/start.sh /usr/local/bin/start.sh
RUN chmod 777 /usr/local/bin/start.sh

COPY ${REQUIREMENTS_FILE_PATH} ${REQUIREMENTS_FILE_PATH}
RUN conda env update --file ${REQUIREMENTS_FILE_PATH} --name base

USER jovyan