FROM python:3.11.12-slim

ARG DEBIAN_FRONTEND="noninteractive"

ARG NON_ROOT_USER="omnianswer"
ARG NON_ROOT_UID="2222"
ARG NON_ROOT_GID="2222"
ARG HOME_DIR="/home/${NON_ROOT_USER}"

ARG REPO_DIR="."

RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER}

ENV PYTHONIOENCODING=utf8
ENV LC_ALL="C.UTF-8"
ENV PATH="/home/omnianswer/.local/bin:${PATH}"

USER ${NON_ROOT_USER}
WORKDIR ${HOME_DIR}/${REPO_DIR}

# Copy only the requirements file first to leverage Docker cache
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/requirements.txt ./requirements.txt

# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} .