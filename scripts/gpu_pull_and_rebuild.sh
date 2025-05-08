#!/bin/bash
# Dockerfileやrequirements.txtを更新したときに使用

set -e

cd ~/ML_COMPETITIONS
git pull origin main
sudo docker build -t my-gpu-app .
