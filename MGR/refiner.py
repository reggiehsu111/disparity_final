import numpy as np
from cv2.ximgproc import weightedMedianFilter
import cv2

def parse_from_refiner(parser):
    return parser


class refiner():
    # Refine the disparity map
    def __init__(self, args):
        self.args = args
        self.max_disp = args.max_disp
        return

    def run(self, D_l, D_r, *args, base=False):
        """
            Params:
                D_l shape (h, w) : Left disparity map
                D_r shape (h, w) : Right disparity map
                *args            : Anything to pass in
                base             : if bas method is used 
            Output shape: (h, w)
        """
        assert(len(D_l.shape)==2 and len(D_r.shape)==2)
        if base:
            out_img = self.base_method(D_l, D_r, *args)
        else:
            # Write you method here
            h, w = D_l.shape
            # Change this line of code
            out_img = np.zeros((h,w)).astype(np.float32)
        assert(len(out_img.shape)==2)
        return out_img

    def base_method(self, D_l, D_r, *args):
        """
            base method used in HW4
        """
        Il = args[0]
        h, w = D_l.shape
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
        return labels.astype(np.float32)

    def print_v(self, message):
        if self.args.verbose:
            print(message)
        return