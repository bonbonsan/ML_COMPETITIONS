#!/bin/bash
# 実行例: ./start_paperspace.sh

source $(dirname "$0")/.paperspace.env

REMOTE_BASE="paperspace@${PAPERSPACE_PUBLIC_IP}:~/ML_COMPETITIONS/my_library"
LOCAL_BASE="$(pwd)/my_library"

ssh -L 8888:localhost:8888 paperspace@$PAPERSPACE_PUBLIC_IP
