#!/bin/bash

source $(dirname "$0")/.paperspace.env

REMOTE_BASE="paperspace@${PAPERSPACE_PUBLIC_IP}:~/ML_COMPETITIONS/my_library"
LOCAL_BASE="$(pwd)/my_library"

mkdir -p $LOCAL_BASE/data $LOCAL_BASE/log $LOCAL_BASE/output

rsync -av --progress $REMOTE_BASE/data/ $LOCAL_BASE/data/
rsync -av --progress $REMOTE_BASE/log/ $LOCAL_BASE/log/
rsync -av --progress $REMOTE_BASE/output/ $LOCAL_BASE/output/
