import numpy as np
import cv2
import time
import os.path
from utils.util import writePFM, readPFM, cal_avgerr
import sys
import json
from MGR.disp import *
from MGR.optimizer import *
from MGR.refiner import *
from MGR.costmgr import *
from utils.parser import ConfigParser

# parser = argparse.ArgumentParser(description='Disparity Estimation')
parser = ConfigParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='../../data/Synthetic/TL3.png', type=str, help='input left image')
parser.add_argument('--input-right', default='../../data/Synthetic/TR3.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL3.pfm', type=str, help='left disparity map')
parser.add_argument('--GT', type=str, help='path to the ground truth image')
parser.add_argument('--max_disp', default=60, type=int, help='maximum disparity possible')
parser.add_argument('--verbose', action='store_true', help='specify if you want to print out verbosely')
parser.add_argument('-c','--config', action='store_true', help='specify if you want to read additional arguments from config')

parser = parse_from_disp(parser)
parser = parse_from_optimizer(parser)
parser = parse_from_refiner(parser)
parser = parse_from_costmgr(parser)

def main():

    args = parser.parse_args()
    # If reading from config:
    if args.config:
        parser.read_config('Config.json')
    parser.print_config()
    # Change the output_path if needed
    parser.write_config('Config.json')
    # Also keep a copy here
    with open('log/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    #add
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