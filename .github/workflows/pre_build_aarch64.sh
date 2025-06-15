#!/bin/bash

dpkg --add-architecture arm64

# add arm64 source
cat <<'EOF'>> /etc/apt/sources.list.d/arm64.list
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ jammy main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ jammy-backports main restricted universe multiverse
EOF

# specifcy amd64 source in the main source.list
sed -i -e 's/deb mirror/deb [arch=amd64] mirror/g' /etc/apt/sources.list
sed -i -e 's/deb http/deb [arch=amd64] http/g' /etc/apt/sources.list

# update and install deps
apt update
apt install -y binutils-aarch64-linux-gnu clang gcc-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu libasound2-dev libcrypt-dev libnsl-dev libxv-dev \
  multistrap pkg-config python3-pip rpcsvc-proto xz-utils

pip install "maturin>=1.3,<2.0" patchelf

# # download ffmpeg archive and extract it
# curl -L "$FFMPEG_DOWNLOAD_URL" -o ffmpeg.tar.xz
# mkdir -p $FFMPEG_DIR
# tar -xf ffmpeg.tar.xz -C $FFMPEG_DIR --strip-components=1
# export FFMPEG_DIR=$FFMPEG_DIR
