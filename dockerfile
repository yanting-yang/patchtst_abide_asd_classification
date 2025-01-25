ARG BASE_IMAGE=ubuntu:jammy-20240911.1
ARG PYTHON_VERSION=3.11

FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

FROM dev-base AS conda
RUN curl -L -o miniconda.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

FROM conda AS conda-install
ARG PYTHON_VERSION
RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} gh && \
    /opt/conda/bin/conda clean -ya

FROM conda-install AS pip-install
COPY requirements.txt .
RUN /opt/conda/bin/pip install -r requirements.txt && \
    rm requirements.txt && \
    /opt/conda/bin/conda clean -ya

FROM pip-install AS dev
LABEL "org.opencontainers.image.source"="https://github.com/yanting-yang/fmri_clip"
ENV PATH=/opt/conda/bin:$PATH
