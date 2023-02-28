#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0

DAVAROCR_PATH=/home/jupyter/kweston_persistent_workspace/DAVAR-Lab-OCR

python $DAVAROCR_PATH/tools/test.py  ./config/publaynet_x101.py ./log/publaynet_x101/Best_checkpoint.pth --eval bbox --show-dir showdir