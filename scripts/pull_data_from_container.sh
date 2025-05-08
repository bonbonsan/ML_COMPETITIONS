#!/bin/bash

# 自分のマシンのIPに変更
PAPERSPACE_PUBLIC_IP="xxx.xxx.x.xxx"
# PAPERSPACE_PUBLIC_IP="184.105.4.230" 

REMOTE_BASE="paperspace@$PAPERSPACE_PUBLIC_IP:~/ML_COMPETITIONS/my_library"
LOCAL_BASE="$(pwd)/my_library"

mkdir -p $LOCAL_BASE/data $LOCAL_BASE/log $LOCAL_BASE/output

scp -r $REMOTE_BASE/data/* $LOCAL_BASE/data/
scp -r $REMOTE_BASE/log/* $LOCAL_BASE/log/
scp -r $REMOTE_BASE/output/* $LOCAL_BASE/output/
