# Taken from (https://catalog.redhat.com/software/containers/rhel9/python-311/63f764969b0ca19f84f7e7c0)
ARG BASE_REGISTRY=registry.redhat.io
ARG BASE_IMAGE=rhel9/python-311
ARG BASE_TAG=1-77.1726696860

################
# App Base
# Installs and sets up poetry environment variables
################
FROM ${BASE_REGISTRY}/${BASE_IMAGE}:${BASE_TAG} AS base

ENV APP_ROOT=/opt/app-root \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TZ=Asia/Singapore \
    CNB_STACK_ID=com.redhat.stacks.ubi9-python-311 \
    CNB_USER_ID=1001 \
    CNB_GROUP_ID=0 \
    POETRY_REQUESTS_TIMEOUT=300 \
    POETRY_VERSION=1.8.3 \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # this is where our requirements + virtual environment will live
    VENV_PATH="$APP_ROOT/.venv"

# prepend venv to path
ENV PATH="$VENV_PATH/bin:$PATH"

RUN chown -R 1001:0 $APP_ROOT \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir poetry==$POETRY_VERSION

# copy project requirement files here to ensure they will be cached.
WORKDIR $APP_ROOT
COPY --chown=1001:0 poetry.lock pyproject.toml ./

################
# Development
# Sets up environment for code development
################

FROM base AS development

COPY docker-scripts/ /usr/bin

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# --no-root is used to just install dependencies as development code will be mounted

# Install libsndfile1 (linux soundfile package)
# RUN apt-get clean \
#     && apt-get update \ 
#     && apt-get install -y gcc g++ libsndfile1 ffmpeg sox wget git \
#     && rm -rf /var/lib/apt/lists/*

RUN poetry install --no-root \
    && rm -rf $HOME/.cache/pypoetry/artifacts \
    && rm -rf $HOME/.cache/pypoetry/cache \
    # Poetry creates folders that requires permission fixes
    && fix-permissions ${APP_ROOT} -P \
    && rpm-file-permissions

ARG NEMO_VERSION=1.23.0
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir Cython==0.29.35 && \
    pip3 install --no-cache-dir nemo_toolkit[asr]==${NEMO_VERSION}

# The following echo adds the unset command for the variables set below to the \
# venv activation script. This is inspired from scl_enable script and prevents \
# the virtual environment to be activated multiple times and also every time \
# the prompt is rendered.
RUN echo "unset BASH_ENV PROMPT_COMMAND ENV" >> $VENV_PATH/bin/activate

USER 1001

WORKDIR /opt/app-root

# NOTE: Only uncomment when making the final .tar file
# ADD /pretrained_models /opt/app-root/pretrained_models
# ADD /asr_inference_service /opt/app-root/asr_inference_service

# For RHEL/Centos 8+ scl_enable isn't sourced automatically in s2i-core
# so virtualenv needs to be activated this way
ENV BASH_ENV="$VENV_PATH/bin/activate" \
    ENV="$VENV_PATH/bin/activate" \
    PROMPT_COMMAND=". $VENV_PATH/bin/activate"


#RUN ["python", "-c", "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; NeuralDiarizer.from_pretrained('diar_msdd_telephonic')"]
RUN ["python", "-c", "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token='HF_TOKEN_HERE')"]

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
#RUN ["python", "-c", "from denoiser import pretrained; pretrained.dns64()"]