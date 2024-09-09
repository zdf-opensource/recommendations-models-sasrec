# Copyright (c) 2024, ZDF.
# Default base image.
ARG BASE_IMAGE

FROM $BASE_IMAGE

#ARG PA_BASE_PACKAGE_URL
#ARG PA_BASE_PACKAGE_VERSION

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

#PA Base installed within poetry, ignored within docker

ADD docker/code/* /opt/program/

WORKDIR /opt/program/


ENTRYPOINT ["python"]
