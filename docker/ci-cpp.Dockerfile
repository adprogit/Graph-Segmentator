# Image CI pour la compilation et les tests C++ (voir .gitlab-ci.yml)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        build-essential \
        clang-format \
        cmake \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
