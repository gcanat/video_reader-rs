#!/bin/bash

# update and install deps
dnf -y update
dnf install -y binutils clang gcc alsa-lib-devel libxcrypt-devel libnsl \
    libXv-devel python3-pip rpcsvc-proto-devel xz uuid-devel libxcb

pip install "maturin>=1.3,<2.0" patchelf
