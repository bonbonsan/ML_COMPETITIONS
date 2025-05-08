#!/bin/bash
cd ~/ML_COMPETITIONS
git pull origin main
sudo docker run --gpus all -p 8888:8888 -it --env-file .env -v $(pwd):/workspace my-gpu-app
