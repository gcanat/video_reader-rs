#!/bin/bash
dnf install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm -y
dnf install --nogpgcheck https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-$(rpm -E %rhel).noarch.rpm -y
dnf install -y clang clang-devel clang-libs ffmpeg-libs glibc-devel libXv-devel xz alsa-lib-devel
python3 -m ensurepip && pip install "maturin>=1.3,<2.0" patchelf
