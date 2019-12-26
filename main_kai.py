import numpy as np
import argparse
import cv2
import time
from util import writePFM, readPFM, cal_avgerr
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from scipy.ndimage import gaussian_filter
from cv2.ximgproc import guidedFilter, weightedMedianFilter

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    cost_volume_l = np.zeros((max_disp+1, h, w))
    cost_volume_r = np.zeros((max_disp+1, h, w))

    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    for d in range(0, max_disp+1):
        Ir_d = Ir[:, :w-d, :]
        Il_part = Il[:, d:, :]
        cost_SD_l = np.abs(Il_part - Ir_d).sum(axis=2)
        cost_SD_l[cost_SD_l > 255] = 255

        # Il_d = Il[:, d:, :]
        Ir_part = Ir[:, :w-d, :]
        # cost_SD_r = np.abs(Ir_part - Il_d).sum(axis=2)  # actually same as cost_SD_l
        # cost_SD_r[cost_SD_r > 255] = 255

        # >>> Cost aggregation
        # TODO: Refine cost by aggregate nearby costs
        cost_SD_l_filtered = guidedFilter(guide=Il_part.astype(np.uint8),
                                          src=cost_SD_l.astype(np.uint8), radius=9, eps=4, dDepth=-1)
        cost_volume_l[d, :, :] = np.concatenate((np.full((h, d), 999), cost_SD_l_filtered), axis=1)

        cost_SD_r_filtered = guidedFilter(guide=Ir_part.astype(np.uint8),
                                          src=cost_SD_l.astype(np.uint8), radius=9, eps=4, dDepth=-1)
        cost_volume_r[d, :, :] = np.concatenate((cost_SD_r_filtered, np.full((h, d), 999)), axis=1)

    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    D_l = cost_volume_l.argmin(axis=0)
    D_r = cost_volume_r.argmin(axis=0)

    # return D_l.astype(np.uint8)

    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    # consistency check
    y, x = np.indices((h, w))
    check_idx = D_l == D_r[y, np.maximum(x-D_l[y, x], 0)]
    # create F_l
    F_l = np.where(check_idx, D_l, np.full((h, w), np.nan))
    fill_vector = np.full((h,), np.inf)
    for j in range(w):
        col = F_l[:, j]
        fill_vector = np.where(~np.isnan(col), col, fill_vector)
        F_l[:, j] = np.where(np.isnan(col), fill_vector, col)

    # compute F_r
    F_r = np.where(check_idx, D_l, np.full((h, w), np.nan))
    fill_vector = np.full((h,), np.inf)
    for j in range(w)[::-1]:
        col = F_r[:, j]
        fill_vector = np.where(~np.isnan(col), col, fill_vector)
        F_r[:, j] = np.where(np.isnan(col), fill_vector, col)

    labels = np.minimum(F_l, F_r)
    labels_filtered = weightedMedianFilter(joint=Il.astype(np.uint8), src=labels.astype(np.uint8), r=32, sigma=15)
    labels = np.where(check_idx, labels, labels_filtered)

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
