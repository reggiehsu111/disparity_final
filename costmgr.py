import numpy as np
from cv2.ximgproc import guidedFilter

def parse_from_costmgr(parser):
    return parser

class costMgr():
    def __init__(self, args):
        self.args = args
        self.max_disp = args.max_disp
        return

    def run(self, in_img_l, in_img_r, *args, base=False):
        """
            Params:
                Input shape  : (h, w, 3)
                *args        : Anything to pass in
                base         : if bas method is used 
            Output shape : (max_disp+1, h, w)
        """
        assert(in_img_l.shape[2] == 3 and in_img_r.shape[2] == 3 )
        if base:
            out_img_l, out_img_r = self.base_method(in_img_l, in_img_r)
        else:
            # Write you method here
            h, w, ch = in_img_l.shape
            # Change this line of code
            out_img_l, out_img_r = np.zeros((self.max_disp, h, w)), np.zeros((self.max_disp, h, w))

        assert(out_img_l.shape[0] == self.args.max_disp+1 and out_img_r.shape[0] == self.args.max_disp+1)
        return out_img_l, out_img_r

    def base_method(self, Il, Ir):
        """
            base method used in HW4
        """
        h, w, ch = Il.shape
        # Array to store disparities for window sliding left, i.e. Ir sliding right
        cost_matrix_left = np.zeros((self.max_disp+1, h, w))
        # Array to store disparities for window sliding right
        cost_matrix_right = np.zeros((self.max_disp+1, h, w))
        # Define padding
        padding=0

        # >>> Cost computation
        # TODO: Compute matching cost from Il and Ir
        # Block matching
        for x in range(self.max_disp+1):
            tmp = np.sum(np.abs(Il[:, x:w] - Ir[:, 0:w-x]), axis = 2)
            tmp = guidedFilter(guide=Il[:,x:w], src=tmp.astype(np.uint8), radius=5, eps=4, dDepth=-1)
            tmp_l = np.hstack((np.full((h, x), padding), tmp))
            tmp_l = np.clip(tmp_l, 0, 255)
            cost_matrix_left[x] = tmp_l
            tmp_r = np.hstack((tmp, np.full((h, x), padding)))
            tmp_r = np.clip(tmp_r, 0, 255)
            cost_matrix_right[x] = tmp_r

        return cost_matrix_left, cost_matrix_right

    def get_cost(self):
        return

    def cost_aggregate(self):
        return