# python3 main.py --input-left <path to left image> --input-right <path toright image> --output <path to output PFM file>
#!/bin/bash
DATA_DIR="data/"
SYN_DIR="Real/"
OUT_DIR="Output/"
LOG_DIR="log/"
LOGFILE="${LOG_DIR}log.txt"
LOGERROR="${LOG_DIR}log_error.txt"
rm $LOGFILE
touch $LOGFILE
rm $LOGERROR
touch $LOGERROR
for i in {0..9}
	do
		LEFT_PATH="${DATA_DIR}${SYN_DIR}TL$i.bmp"
		RIGHT_PATH="${DATA_DIR}${SYN_DIR}TR$i.bmp"
		OUT_PATH="${DATA_DIR}${OUT_DIR}real_out$i.pfm"
		touch $OUT_PATH
		echo -e "\n\nlogging photo $i ...\n" | tee -a $LOGFILE
		python main.py --input-left $LEFT_PATH --input-right $RIGHT_PATH --output $OUT_PATH --N 256 --real| tee -a $LOGFILE
	done
# python3 utils/errors.py | tee -a $LOGFILE