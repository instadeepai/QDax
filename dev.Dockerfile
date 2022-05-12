FROM python:3.8.13-bullseye as venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_FOLDER=/app
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR $APP_FOLDER

COPY qdax qdax
COPY setup.py ./

RUN pip install .


FROM venv-image as test-image

ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt


FROM venv-image as venv-cuda

RUN pip --no-cache-dir install jaxlib==0.3.2+cuda11.cudnn82 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as dev-image
# The dev-image does not contain the any file, qdax is expected to be mounted
# afterwards

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

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
    python3=3.8* \
    python3-pip \
    screen \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    nano \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH
RUN ln -s /usr/bin/python3 /usr/local/bin/python

COPY --from=venv-cuda /opt/venv/. /opt/venv/

COPY requirements-dev.txt ./
RUN pip --no-cache-dir install -r requirements-dev.txt && \
    pip install pyopengl && \
    rm -rf /tmp/*
USER $USER


FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04 as run-image-cuda

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:$PATH
RUN ln -s /usr/bin/python3 /usr/local/bin/python

COPY --from=venv-cuda /opt/venv/. /opt/venv/

CMD ["python"]

FROM python:3.8.13-slim-bullseye as run-image

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PATH=/opt/venv/bin:$PATH

COPY --from=venv-image /opt/venv/. /opt/venv/

CMD ["python"]
