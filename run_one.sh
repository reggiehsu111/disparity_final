#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Synthetic/"
OUT_DIR="Output/"
i=0
LEFT_PATH="${DATA_DIR}${SYN_DIR}TL$i.png"
RIGHT_PATH="${DATA_DIR}${SYN_DIR}TR$i.png"
GT_PATH="${DATA_DIR}${SYN_DIR}TLD$i.pfm"
OUT_PATH="${DATA_DIR}${OUT_DIR}out$i.pfm"
touch $OUT_PATH
python3 main.py --input-left $LEFT_PATH --input-right $RIGHT_PATH --output $OUT_PATH --GT $GT_PATH
# Visualize the result
# Uncomment to visualize result
#. vis1.sh $i