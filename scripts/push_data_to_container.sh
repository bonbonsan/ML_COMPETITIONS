#!/bin/bash
# MacローカルのデータをPaperspaceに送信

source $(dirname "$0")/.paperspace.env

REMOTE_BASE="paperspace@${PAPERSPACE_PUBLIC_IP}:~/ML_COMPETITIONS/my_library"
LOCAL_BASE="$(pwd)/my_library"

rsync -av --progress $LOCAL_BASE/data/ $REMOTE_BASE/data/
rsync -av --progress $LOCAL_BASE/input/ $REMOTE_BASE/input/
