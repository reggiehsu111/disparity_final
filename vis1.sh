#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Synthetic/"
OUT_DIR="Output/"
i=$1
VIS_PATH="${DATA_DIR}${OUT_DIR}TL2.pfm"
GT_PATH="${DATA_DIR}${SYN_DIR}TLD$i.pfm"
python utils/visualize.py $VIS_PATH $GT_PATH