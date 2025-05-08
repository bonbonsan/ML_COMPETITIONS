#!/bin/bash
# 実行例: ./start_paperspace.sh

# 自分のマシンのIPに変更
PAPERSPACE_PUBLIC_IP="xxx.xxx.x.xxx"
# PAPERSPACE_PUBLIC_IP="184.105.4.230" 
ssh -L 8888:localhost:8888 paperspace@$PUBLIC_IP
