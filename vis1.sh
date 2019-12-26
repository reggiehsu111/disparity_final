#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Synthetic/"
OUT_DIR="Output/"
i=$1
VIS_PATH="${DATA_DIR}${OUT_DIR}out$i.pfm"
GT_PATH="${DATA_DIR}${SYN_DIR}TLD$i.pfm"
python visualize.py $VIS_PATH $GT_PATH