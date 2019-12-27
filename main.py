import numpy as np
import argparse
import cv2
import time
import os.path
from util import writePFM, readPFM, cal_avgerr
import sys
import json
from disp import *
from optimizer import *
from refiner import *
from costmgr import *
from parser import ConfigParser

# parser = argparse.ArgumentParser(description='Disparity Estimation')
parser = ConfigParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='../../data/Synthetic/TL3.png', type=str, help='input left image')
parser.add_argument('--input-right', default='../../data/Synthetic/TR3.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL3.pfm', type=str, help='left disparity map')
parser.add_argument('--GT')
parser.add_argument('--max_disp', default=60, type=int, help='maximum disparity possible')
parser.add_argument('--verbose', action='store_true', help='specify if you want to print out verbosely')

parser = parse_from_disp(parser)
parser = parse_from_optimizer(parser)
parser = parse_from_refiner(parser)
parser = parse_from_costmgr(parser)

def main():
    
    args = parser.parse_args()
    parser.print_config()
    parser.write_config('Config.json')
    with open('log/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    DM = dispMgr(args)
    disp = DM.computeDisp(img_left,img_right)
    # Only when GT is valid
    if args.GT is not None:
        GT = readPFM(args.GT)
        error = cal_avgerr(GT,disp)
        print("Error is:", error)
        # append error to log_error.txt
        with open("log/log_error.txt","a") as f:
            f.write(str(error)+'\n')
    

    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))




if __name__ == '__main__':
    main()
