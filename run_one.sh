#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Synthetic/"
OUT_DIR="Output/"
i=3
LEFT_PATH="${DATA_DIR}${SYN_DIR}TL$i.png"
RIGHT_PATH="${DATA_DIR}${SYN_DIR}TR$i.png"
GT_PATH="${DATA_DIR}${SYN_DIR}TLD$i.pfm"
OUT_PATH="${DATA_DIR}${OUT_DIR}out$i.jpg"
touch $OUT_PATH
# If reading from config:
python main.py -c

# Else specify arguments:
# python3 main.py --input-left $LEFT_PATH --input-right $RIGHT_PATH --output $OUT_PATH --GT $GT_PATH --CM_base --OP_base --RF_base

# Visualize the result:
# Uncomment to visualize result
#. vis1.sh $i