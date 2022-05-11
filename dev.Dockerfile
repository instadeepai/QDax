FROM mambaorg/micromamba:0.22.0 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV MAMBA_ROOT_PREFIX="/opt/conda"

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY environment.yaml /tmp/environment.yaml


RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete


FROM debian:bullseye-20190708 as test-image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=/opt/conda/envs/qdaxpy38/bin/:$PATH
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
COPY requirements-dev.txt ./

RUN pip install -r requirements-dev.txt



FROM conda as conda-cuda

RUN pip --no-cache-dir install jaxlib==0.3.2+cuda11.cudnn82 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    && rm -rf /tmp/*


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as dev-image
# The dev-image does not contain the any file, qdax is expected to be mounted
# afterwards

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib
ENV PATH=/opt/conda/envs/qdaxpy38/bin/:$PATH

ENV DISTRO ubuntu2004
ENV CPU_ARCH x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$CPU_ARCH/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    screen \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    nano \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

COPY --from=conda-cuda /opt/conda/envs/. /opt/conda/envs/

COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip --no-cache-dir install -r /tmp/requirements-dev.txt && \
    pip install pyopengl && \
    rm -rf /tmp/*
USER $USER


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as run-image-cuda

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=/opt/conda/envs/qdaxpy38/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY --from=conda-cuda /opt/conda/envs/. /opt/conda/envs/

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
RUN groupadd --gid ${GROUP_ID} $GROUP && useradd -g $GROUP --uid ${USER_ID} --shell /usr/sbin/nologin -m $USER  && chown -R $USER:$GROUP $APP_FOLDER
USER $USER

COPY qdax qdax
COPY setup.py ./

RUN pip install .

WORKDIR /

CMD ["python"]

FROM debian:bullseye-20190708 as run-image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=/opt/conda/envs/qdaxpy38/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
RUN groupadd --gid ${GROUP_ID} $GROUP && useradd -g $GROUP --uid ${USER_ID} --shell /usr/sbin/nologin -m $USER  && chown -R $USER:$GROUP $APP_FOLDER
USER $USER

COPY qdax qdax
COPY setup.py ./

RUN pip install .

WORKDIR /

CMD ["python"]
