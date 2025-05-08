#!/bin/bash
# MacローカルのデータをPaperspaceに送信

# 自分のマシンのIPに変更
PAPERSPACE_PUBLIC_IP="xxx.xxx.x.xxx"
# PAPERSPACE_PUBLIC_IP="184.105.4.230" 

REMOTE_BASE="paperspace@${PAPERSPACE_PUBLIC_IP}:~/ML_COMPETITIONS/my_library"
LOCAL_BASE="$(pwd)/my_library"

scp -r $LOCAL_BASE/data $REMOTE_BASE/
scp -r $LOCAL_BASE/input $REMOTE_BASE/
