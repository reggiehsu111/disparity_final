import numpy as np
import argparse
import cv2
import time
import os.path
from util import writePFM, readPFM, cal_avgerr
from disp import dispMgr
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from scipy.ndimage import gaussian_filter
from cv2.ximgproc import guidedFilter
from optimizer import *
from refiner import *
from costmgr import *

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='../../data/Synthetic/TL3.png', type=str, help='input left image')
parser.add_argument('--input-right', default='../../data/Synthetic/TR3.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL3.pfm', type=str, help='left disparity map')
parser.add_argument("--GT")

parser = parse_from_optimizer(parser)
parser = parse_from_refiner(parser)
parser = parse_from_costmgr(parser)

def main():
    
    args = parser.parse_args()

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
        with open("log_error.txt","w") as f:
            f.write(error)
        
    #cv2.imwrite('outlier/' + os.path.split(args.output)[1][:-3] + 'png', outlier)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))




if __name__ == '__main__':
    main()
