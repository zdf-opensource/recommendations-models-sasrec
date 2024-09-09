# Copyright (c) 2024, ZDF.
# Build using the following command:
#  docker build -f docker-base/Dockerfile --build-arg "GITLAB_USERNAME=${GITLAB_USERNAME}" --build-arg "GITLAB_ACCESS_TOKEN=${GITLAB_ACCESS_TOKEN}" .
FROM tensorflow/tensorflow:2.10.0-gpu

# Provide gitlab username and access token so that we can install the pa-base pip package.
ARG GITLAB_USERNAME
ARG GITLAB_ACCESS_TOKEN

# Set working directory.
WORKDIR /opt/program/

# Add poetry depedencies and lock file.
ADD pyproject.toml /opt/program/
ADD poetry.lock /opt/program/

# Install system dependencies.
RUN apt-get -y update && apt-get install -y --no-install-recommends nginx

# Update the pip version
RUN python -m pip install --upgrade pip

# Install poetry.
RUN pip install poetry==1.7.1 poetry-plugin-export==1.6.0
RUN poetry config virtualenvs.create false
# Create dummy README so that poetry does not break, since it expects a README file.
RUN touch README.md
# Configure gitlab access token.
RUN poetry config http-basic.pa-base $GITLAB_USERNAME $GITLAB_ACCESS_TOKEN
# Install environment dependencies using pip since poetry errors out on PEP-517 incompatible packages (lightfm, scikit-surprise, recommenders).
# RUN poetry install --without dev --no-root
RUN poetry export -f requirements.txt --without dev --without-hashes --with-credentials | pip install -r /dev/stdin

# # Clear pip and poetry cache.
RUN rm -rf ~/.cache/*
