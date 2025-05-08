#!/bin/bash
# 初回セットアップ：NVIDIAドライバ、nvidia-docker2、プロジェクト取得、Docker初回ビルド

set -e

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-docker2
sudo systemctl restart docker

nvidia-smi

git clone https://github.com/bonbonsan/ML_COMPETITIONS.git
cd ML_COMPETITIONS
cp .env.example .env
sed -i 's/USE_GPU=False/USE_GPU=True/' .env

sudo docker build -t my-gpu-app .
