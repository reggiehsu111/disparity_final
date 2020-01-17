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
parser.add_argument('-r','--real', action='store_true', help='specify if the input images are real')

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
    REAL = False
    if args.input_left.endswith("bmp"):
        REAL = True
    args.real = (args.real or REAL)   
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    # max distance
    print(args.max_disp)
    if REAL:
        args.max_disp = max_dis_real(img_left, img_right)
        args.CM_base = False
    else:
        args.max_disp = max_dis(img_left, img_right)
        args.CM_base = True
    print(args.max_disp)
    #add hisEqulColor
    img_left = hisEqulColor(img_left)
    img_right = hisEqulColor(img_right)
    DM = dispMgr(args)
    disp = DM.computeDisp(img_left,img_right)
    # cv2.imwrite("data/Output/out"+args.output.split('/')[1].split('.')[0]+".jpg", disp)
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

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def max_dis(img_left, img_right):
    surf = cv2.xfeatures2d.SURF_create(1000)
    bf = cv2.BFMatcher()
    left_kp, left_des = surf.detectAndCompute(img_left,None)
    right_kp, right_des = surf.detectAndCompute(img_right,None)
    matches = bf.knnMatch(left_des, right_des, k=2)
    good = list()
    pos = 0
    for (m, n) in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    dis = list()
    for _m in good:
        left_idx = left_kp[_m.queryIdx].pt
        right_idx = right_kp[_m.trainIdx].pt
        if (left_idx[0] > right_idx[0]):
            dis.append(left_idx[0] - right_idx[0])
    dis = (np.array(dis))
    dis = np.abs(dis)
    dis = np.sort(dis)
    max_dis = 60
    for i in range(len(dis)):
        if (dis[len(dis)-1-i] - dis[len(dis)-2-i]) < 2 and (dis[len(dis)-1-i] - dis[len(dis)-3-i]) < 2:
            max_dis = dis[len(dis)-1-i]
            break
    max_dis += 10
    return int(max_dis)

def max_dis_real(img_left, img_right):
    surf = cv2.xfeatures2d.SURF_create(1000)
    bf = cv2.BFMatcher()
    left_kp, left_des = surf.detectAndCompute(img_left,None)
    right_kp, right_des = surf.detectAndCompute(img_right,None)
    matches = bf.knnMatch(left_des, right_des, k=2)
    good = list()
    pos = 0
    for (m, n) in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    dis = list()
    for _m in good:
        left_idx = left_kp[_m.queryIdx].pt
        right_idx = right_kp[_m.trainIdx].pt
        if (left_idx[0] > right_idx[0]):
            dis.append(left_idx[0] - right_idx[0])
    dis = (np.array(dis))
    dis = np.abs(dis)
    dis = np.sort(dis)
    max_dis = 60
    for i in range(len(dis)):
        if (dis[len(dis)-1-i] - dis[len(dis)-2-i]) < 2 and (dis[len(dis)-1-i] - dis[len(dis)-3-i]) < 2:
            max_dis = dis[len(dis)-1-i]
            break
    max_dis += 1
    temp = max(6,int(max_dis))
    temp = min(temp, 70)
    return int(temp)

if __name__ == '__main__':
    main()