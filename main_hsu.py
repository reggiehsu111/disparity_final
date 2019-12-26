import numpy as np
import argparse
import cv2
import time
from util import writePFM, readPFM, cal_avgerr
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from scipy.ndimage import gaussian_filter
from cv2.ximgproc import guidedFilter

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # Array to store disparities for window sliding left, i.e. Ir sliding right
    cost_matrix_left = np.zeros((max_disp+1, h, w))
    # Array to store disparities for window sliding right
    cost_matrix_right = np.zeros((max_disp+1, h, w))
    # Define padding
    padding=0

    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    # Block matching
    for x in range(max_disp+1):
        tmp = np.sum(np.abs(Il[:, x:w] - Ir[:, 0:w-x]), axis = 2)
        tmp = cv2.ximgproc.guidedFilter(guide=Il[:,x:w], src=tmp.astype(np.uint8), radius=5, eps=4, dDepth=-1)
        tmp_l = np.hstack((np.full((h, x), padding), tmp))
        tmp_l = np.clip(tmp_l, 0, 255)
        cost_matrix_left[x] = tmp_l
        tmp_r = np.hstack((tmp, np.full((h, x), padding)))
        tmp_r = np.clip(tmp_r, 0, 255)
        cost_matrix_right[x] = tmp_r


  
    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    # cost_matrix_left = cv2.ximgproc.guidedFilter(
    #         guide=Il, src=np.moveaxis(cost_matrix_left.astype(np.uint8),0,2), radius=5, eps=5, dDepth=-1)

    # cost_matrix_right = cv2.ximgproc.guidedFilter(
    #         guide=Ir, src=np.moveaxis(cost_matrix_right.astype(np.uint8),0,2), radius=5, eps=5, dDepth=-1)
    # cost_matrix_left = np.rollaxis(cost_matrix_left,2,0)
    # cost_matrix_right = np.rollaxis(cost_matrix_right,2,0)

    
    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.

    displ = np.argmin(cost_matrix_left, axis=0)
    dispr = np.argmin(cost_matrix_right, axis=0)

    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    # Left-right consistency check
    FL = displ.astype(np.float32)
    FR = displ.astype(np.float32)
    occlusion = np.zeros((h,w))
    for x in range(h):
        for y in range(w):
            occlusion[x,y] = abs(displ[x,y] - dispr[x, y-displ[x,y]])>1
    occlusion = occlusion.astype(np.uint8)
    # cv2.imwrite("occ.png", np.uint8(occlusion*255))
    FL[np.where(occlusion[:,0]==1),0] = np.inf
    FR[np.where(occlusion[:,w-1]==1),w-1] = np.inf
    for x in range(1,w-1):
        FL[np.where(occlusion[:,x]==1),x] = FL[np.where(occlusion[:,x]==1),x-1]
        FR[np.where(occlusion[:,w-x-1]==1),w-x-1] = FR[np.where(occlusion[:,w-x-1]==1),w-x]
    # cv2.imwrite("Disp.png",np.uint8(displ))
    # cv2.imwrite("FL.png", np.uint8(FL))
    # cv2.imwrite("FR.png", np.uint8(FR))
    threshold = max_disp//6
    FL = np.clip(FL,threshold,255)
    FR = np.clip(FR,threshold,255)
    labels = np.maximum(FL,FR)
    # labels=displ
    wmf = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), r=55, sigma=7)
    labels[np.where(occlusion==1)] = wmf[np.where(occlusion==1)]
    wmf = cv2.ximgproc.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), r=25, sigma=7)
    labels = wmf

    # labels = wmf
    labels = cv2.medianBlur(np.float32(labels),5)
    labels = cv2.bilateralFilter(labels, 5, 1, 7)

    return labels.astype(np.uint8)


def main():
    parser.add_argument("--GT")
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    # Only when GT is valid
    if args.GT is not None:
        GT = readPFM(args.GT)
        print("Error is:", cal_avgerr(GT, disp))
    toc = time.time()
    writePFM(args.output, disp.astype(np.float32))
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    max_disp = 60
    main()
