# python3 main.py --input-left <path to left image> --input-right <path toright image> --output <path to output PFM file>
#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Synthetic/"
OUT_DIR="Output/"
LOG_DIR="log/"
LOGFILE="log.txt"
rm $LOGFILE
touch $LOGFILE
for i in {0..9}
	do
		LEFT_PATH="${DATA_DIR}${SYN_DIR}TL$i.png"
		RIGHT_PATH="${DATA_DIR}${SYN_DIR}TR$i.png"
		GT_PATH="${DATA_DIR}${SYN_DIR}TLD$i.pfm"
		OUT_PATH="${DATA_DIR}${OUT_DIR}out$i.pfm"
		touch $OUT_PATH
		echo -e "\n\nlogging photo $i ...\n" | tee -a $LOGFILE
		python3 main.py --input-left $LEFT_PATH --input-right $RIGHT_PATH --output $OUT_PATH --GT $GT_PATH | tee -a $LOGFILE
	done