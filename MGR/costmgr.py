import argparse
import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

WINDOW_SIZE = 37

def parse_from_costmgr(parser):
    parser.add_argument('--arms_th', default=10, type=float, help='the threshold for computing arms')
    parser.add_argument('--log_disp', action='store_true', help='Specify to log all levels of disparity map')
    parser.add_argument('--N', default=256, type=int, help='For the usage of binary stereo matching')
    return parser


def aggregate(costl, costr, phi, axis=2):
    tmp = np.bitwise_xor(costl, costr)
    tmp = np.bitwise_and(tmp, phi)
    tmp = np.sum(tmp, axis=axis)
    return tmp

def compute_cost(w, h, Il_gray, Il_lab, Ir_gray, Ir_lab, N):
    SIGMA = 6
    dx1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dx2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dx1 = np.tile(dx1.reshape((1,1,N)), (h,w,1))
    dx2 = np.tile(dx2.reshape((1,1,N)), (h,w,1))
    dy1 = np.tile(dy1.reshape((1,1,N)), (h,w,1))
    dy2 = np.tile(dy2.reshape((1,1,N)), (h,w,1))

    costl = np.zeros((h, w, N), dtype=np.bool)
    costr = np.zeros((h, w, N), dtype=np.bool)
    phi_l = np.zeros((h, w, N), dtype=np.bool)
    phi_r = np.zeros((h, w, N), dtype=np.bool)
    xx, yy = np.meshgrid(np.linspace(0,w-1, w), np.linspace(0, h-1, h))

    px = np.tile(xx.astype(np.uint16).reshape((h,w,1)), N)
    lower_x = np.clip(px-WINDOW_SIZE // 2, 0, w-1)
    upper_x = np.clip(px+WINDOW_SIZE // 2, 0, w-1)
    px = np.clip(px + dx1, lower_x, upper_x)
    qx = np.clip(px + dx2, lower_x, upper_x)
    
    py = np.tile(yy.astype(np.uint16).reshape((h,w,1)), N)
    lower_y = np.clip(py-WINDOW_SIZE // 2, 0, h-1)
    upper_y = np.clip(py+WINDOW_SIZE // 2, 0, h-1)
    py = np.clip(py + dy1, lower_y, upper_y)
    qy = np.clip(py + dy2, lower_y, upper_y)
    costl = Il_gray[py, px] > Il_gray[qy, qx]
    costr = Ir_gray[py, px] > Ir_gray[qy, qx]
    Il_lab_tile = np.tile(Il_lab.reshape((Il_lab.shape[0], Il_lab.shape[1], 1, Il_lab.shape[2])), (1,1,N,1))
    Ir_lab_tile = np.tile(Ir_lab.reshape((Ir_lab.shape[0], Ir_lab.shape[1], 1, Ir_lab.shape[2])), (1,1,N,1))
    Wl = np.amax(
            np.array([
                np.sum(np.abs(Il_lab_tile - Il_lab[py, px]), axis = 3),
                np.sum(np.abs(Il_lab_tile - Il_lab[qy, qx]), axis= 3)
            ]),
            axis=0
        )
    T = np.tile(np.percentile(Wl, 25, 2).reshape((h, w, 1)), N)
    phi_l = (Wl <= T)
    Wr = np.amax(
            np.array([
                np.sum(np.abs(Ir_lab_tile - Ir_lab[py, px]), axis = 3),
                np.sum(np.abs(Ir_lab_tile - Ir_lab[qy, qx]), axis= 3)
            ]),
            axis=0
        )
    T = np.tile(np.percentile(Wr, 25, 2).reshape((h, w, 1)), N)
    phi_r = (Wr <= T)
    return costl, costr, phi_l, phi_r

def show_costs(matrix):
    print("Writing out...")
    for x in tqdm(range(matrix.shape[0])):
        cv2.imwrite("log/disps/"+str(x)+".jpg", matrix[x])
    return
def single_channel_agg(temp, costl, phi):
    window = 1
    sigma1 = 9
    sigma2 = 16
    h, w = temp.shape
    kernel = np.zeros((2*window+1, 2*window+1))
    kernel_temp = np.copy(kernel)
    ret = np.zeros((h,w))

    for x in range(kernel.shape[1]):
        kernel[:,x] = np.exp(-np.abs(x)/sigma1)
    temp = np.pad(temp, window, 'symmetric')
    for x in range(window, w-window):
        for y in range(window, h-window):
            for j in range(-window, window):
                for i in range(-window, window):
                    kernel_temp[j,i] = kernel[j,i] * np.exp(-aggregate(costl[y+j, x], costl[y+j, x+i], phi[y+j, x], axis=0)/sigma2)
            ret[y-window,x-window] = np.sum(kernel_temp*temp[y-window:y+window+1, x-window:x+window+1])

    return ret

def feature_aggregate(d, h, w, cost_matrix_left, cost_matrix_right, costl, costr, phi_l, phi_r):
    window = 2
    sigma1 = 9
    sigma2 = 16

    # P = np.zeros((w,h,2*window+1))
    # PHI = np.zeros((w, w))
    # gamma = np.zeros((h, w, w))
    # for y in range(h):
    #     for z in range(-window, window):
    #         for x in range(window, w-window):
    #             print(z)
    #             P[x, y, z] = np.exp(-abs(z)/sigma1)*np.exp(-aggregate(costl[y, x], costl[y, x+z]))
    # for y in range(window, h-window):
    #     if y == window:
    #         for x in range(window, w-window):
    #             for z in range(-window, window):
    #                 PHI[x+z, x] = np.sum(P[x, y-window:y+window, z])
    #                 PHI[x, x+z] = PHI[x+z, x]

    #         break
    #     for x in range(w):
    #         for i in range(d):
    #             gamma[y, x+i, x] = np.sum(cost_matrix_left[i, y-window:y+window, x])
    # print(gamma)



    return cost_matrix_left, cost_matrix_right




class costMgrBase:
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
        assert (in_img_l.shape[2] == 3 and in_img_r.shape[2] == 3)
        if base:
            out_img_l, out_img_r = self.base_method(in_img_l, in_img_r)
        else:
            # Write you method here
            out_img_l, out_img_r = self.improved_method(in_img_l, in_img_r)
        assert (out_img_l.shape[0] == self.args.max_disp + 1 and out_img_r.shape[0] == self.args.max_disp + 1)
        return out_img_l, out_img_r

    def base_method(self, Il, Ir):
        """
            base method used in HW4
        """
        h, w, ch = Il.shape
        # Array to store disparities for window sliding left, i.e. Ir sliding right
        cost_matrix_left = np.zeros((self.max_disp + 1, h, w))
        # Array to store disparities for window sliding right
        cost_matrix_right = np.zeros((self.max_disp + 1, h, w))
        # Define padding
        padding = self.max_disp

        # >>> Cost computation
        # TODO: Compute matching cost from Il and Ir
        # Block matching
        for x in range(self.max_disp + 1):
            tmp = np.sum(np.abs(Il[:, x:w] - Ir[:, 0:w - x]), axis=2)
            tmp = guidedFilter(guide=Il[:, x:w], src=tmp.astype(np.uint8), radius=5, eps=4, dDepth=-1)
            tmp_l = np.hstack((np.full((h, x), padding), tmp))
            tmp_l = np.clip(tmp_l, 0, 255)
            cost_matrix_left[x] = tmp_l
            tmp_r = np.hstack((tmp, np.full((h, x), padding)))
            tmp_r = np.clip(tmp_r, 0, 255)
            cost_matrix_right[x] = tmp_r
        if self.args.log_disp:
            show_costs(cost_matrix_left)

        return cost_matrix_left, cost_matrix_right

    def improved_method(self, Il, Ir):

        h, w, ch = Il.shape

        Il = cv2.GaussianBlur(Il, (3,3), 2)
        Ir = cv2.GaussianBlur(Ir, (3,3), 2)
        Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
        Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
        Il_lab = cv2.cvtColor(Il.astype(np.uint8), cv2.COLOR_BGR2Lab).astype('float')
        Ir_lab = cv2.cvtColor(Ir.astype(np.uint8), cv2.COLOR_BGR2Lab).astype('float')
        # Array to store disparities for window sliding left, i.e. Ir sliding right
        cost_matrix_left = np.zeros((self.max_disp+1, h, w))
        # Array to store disparities for window sliding right
        cost_matrix_right = np.zeros((self.max_disp+1, h, w))
        N = self.args.N
        
        print("Getting costs...")
        costl, costr, phi_l, phi_r = compute_cost(w, h, Il_gray, Il_lab, Ir_gray, Ir_lab, N)

        print("Aggregating...")
        padding = self.max_disp
        for d in tqdm(range(self.max_disp+1)):
            tmp = np.zeros((h,w-d))
            tmp = aggregate(costl[:, d:w], costr[:, :w-d], phi_l[:, d:w])
            # tmp = single_channel_agg(tmp, costl[:, d:w], phi_l[:, d:w])
            tmp = guidedFilter(guide=Il[:, d:w], src=tmp.astype(np.uint8), radius=1, eps=50, dDepth=-1)
            # tmp = cv2.bilateralFilter(tmp.astype(np.float32), 5, 9, 16)
            tmp_l = np.hstack((np.full((h, d), padding), tmp))
            tmp_l = np.clip(tmp_l, 0, 255)
            cost_matrix_left[d] = tmp_l
            # tmp = aggregate(costl[:, d:w], costr[:, :w-d], phi_r[:, d:w])
            # tmp = single_channel_agg(tmp, costl[:, d:w], phi_r[:, d:w])
            # tmp = guidedFilter(guide=Il[:, d:w], src=tmp.astype(np.uint8), radius=15, eps=100, dDepth=-1)
            tmp_r = np.hstack((tmp, np.full((h, d), padding)))
            tmp_r = np.clip(tmp_r, 0, 255)
            cost_matrix_right[d] = tmp_r
        # cost_matrix_left, cost_matrix_right = feature_aggregate(self.max_disp+1, h, w, cost_matrix_left, cost_matrix_right, costl, costr, phi_l, phi_r)
        if self.args.log_disp:
            show_costs(cost_matrix_left)
     
        return cost_matrix_left, cost_matrix_right



class costMgr(costMgrBase):
    def __init__(self, args):
        super(costMgr, self).__init__(args)
        self.arms_th = args.arms_th
        self.h, self.w = None, None

    def compute_arms(self, img):
        """
        :param img: image to compute arms on
        :return: a dictionary containing 4 masks, corresponding to the arm lengths of the 4 directions
        """
        img = img.astype(np.float)
        h, w, _ = img.shape
        '''compute horizontal arms'''
        h_plus, h_minus = np.zeros((h, w)), np.zeros((h, w))
        d = 1
        mask_p = np.ones((h, w), dtype=bool)
        mask_m = np.ones((h, w), dtype=bool)
        while (np.any(mask_p) or np.any(mask_m)) and d < w:
            diff = np.abs(img[:, :w - d, :] - img[:, d:, :]).max(axis=2)
            # plus
            diff_p = np.concatenate((diff, np.full((h, d), np.inf)), axis=1)
            mask_p = np.logical_and(mask_p, diff_p < self.arms_th)
            h_plus[mask_p] += 1
            # minus
            diff_m = np.concatenate((np.full((h, d), np.inf), diff), axis=1)
            mask_m = np.logical_and(mask_m, diff_m < self.arms_th)
            h_minus[mask_m] += 1
            # increment d
            d += 1

        '''compute vertical arms'''
        v_plus, v_minus = np.zeros((h, w)), np.zeros((h, w))
        d = 1
        mask_p = np.ones((h, w), dtype=bool)
        mask_m = np.ones((h, w), dtype=bool)
        while (np.any(mask_p) or np.any(mask_m)) and d < h:
            diff = np.abs(img[:h - d, :, :] - img[d:, :, :]).max(axis=2)
            # plus
            diff_p = np.concatenate((diff, np.full((d, w), np.inf)), axis=0)
            mask_p = np.logical_and(mask_p, diff_p < self.arms_th)
            v_plus[mask_p] += 1
            # minus
            diff_m = np.concatenate((np.full((d, w), np.inf), diff), axis=0)
            mask_m = np.logical_and(mask_m, diff_m < self.arms_th)
            v_minus[mask_m] += 1
            # increment d
            d += 1

        arms_dict = {
            'h_p': h_plus,
            'h_m': h_minus,
            'v_p': v_plus,
            'v_m': v_minus
        }
        return arms_dict

    def get_U(self, arms_l, arms_r):
        # for I_l
        U_h_p_l, U_h_m_l = np.zeros((self.max_disp + 1, self.h, self.w)), np.zeros((self.max_disp + 1, self.h, self.w))
        U_v_p_l, U_v_m_l = np.zeros((self.max_disp + 1, self.h, self.w)), np.zeros((self.max_disp + 1, self.h, self.w))
        # for I_r
        U_h_p_r, U_h_m_r = np.zeros((self.max_disp + 1, self.h, self.w)), np.zeros((self.max_disp + 1, self.h, self.w))
        U_v_p_r, U_v_m_r = np.zeros((self.max_disp + 1, self.h, self.w)), np.zeros((self.max_disp + 1, self.h, self.w))

        for d in range(self.max_disp + 1):
            # h_p
            part_U = np.minimum(arms_l['h_p'][:, d:], arms_r['h_p'][:, :self.w - d])
            U_h_p_l[d] = np.concatenate((np.zeros((self.h, d)), part_U), axis=1)
            U_h_p_r[d] = np.concatenate((part_U, np.zeros((self.h, d))), axis=1)
            # h_m
            part_U = np.maximum(arms_l['h_m'][:, d:], arms_r['h_m'][:, :self.w - d])
            U_h_m_l[d] = np.concatenate((np.zeros((self.h, d)), part_U), axis=1)
            U_h_m_r[d] = np.concatenate((part_U, np.zeros((self.h, d))), axis=1)
            # v_p
            part_U = np.minimum(arms_l['v_p'][:, d:], arms_r['v_p'][:, :self.w - d])
            U_v_p_l[d] = np.concatenate((np.zeros((self.h, d)), part_U), axis=1)
            U_v_p_r[d] = np.concatenate((part_U, np.zeros((self.h, d))), axis=1)
            # v_m
            part_U = np.maximum(arms_l['v_m'][:, d:], arms_r['v_m'][:, :self.w - d])
            U_v_m_l[d] = np.concatenate((np.zeros((self.h, d)), part_U), axis=1)
            U_v_m_r[d] = np.concatenate((part_U, np.zeros((self.h, d))), axis=1)

        U_dict_l = {
            'h_p': U_h_p_l.astype(int),
            'h_m': U_h_m_l.astype(int),
            'v_p': U_v_p_l.astype(int),
            'v_m': U_v_m_l.astype(int)
        }
        U_dict_r = {
            'h_p': U_h_p_r.astype(int),
            'h_m': U_h_m_r.astype(int),
            'v_p': U_v_p_r.astype(int),
            'v_m': U_v_m_r.astype(int)
        }
        return U_dict_l, U_dict_r

    def get_cost(self, I_l, I_r):
        print("==> Compute base cost...")
        cost_l = np.zeros((self.max_disp + 1, self.h, self.w))
        cost_r = np.zeros((self.max_disp + 1, self.h, self.w))
        for d in range(self.max_disp + 1):
            diff = np.abs(I_r[:, :self.w - d, :] - I_l[:, d:, :]).sum(axis=2)
            diff[diff > 255] = 255
            cost_l[d] = np.concatenate((np.full((self.h, d), 999), diff), axis=1)
            cost_r[d] = np.concatenate((diff, np.full((self.h, d), 999)), axis=1)
        return cost_l, cost_r

    def get_cost_BSM(self, Il, Ir):
        print("==> Compute BSM cost...")

        h, w, ch = Il.shape

        Il = cv2.GaussianBlur(Il, (3, 3), 2)
        Ir = cv2.GaussianBlur(Ir, (3, 3), 2)
        Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
        Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
        Il_lab = cv2.cvtColor(Il.astype(np.uint8), cv2.COLOR_BGR2Lab).astype('float')
        Ir_lab = cv2.cvtColor(Ir.astype(np.uint8), cv2.COLOR_BGR2Lab).astype('float')
        # Array to store disparities for window sliding left, i.e. Ir sliding right
        cost_matrix_left = np.zeros((self.max_disp + 1, h, w))
        # Array to store disparities for window sliding right
        cost_matrix_right = np.zeros((self.max_disp + 1, h, w))
        N = self.args.N

        print("Getting costs...")
        costl, costr, phi_l, phi_r = compute_cost(w, h, Il_gray, Il_lab, Ir_gray, Ir_lab, N)

        print("Aggregating...")
        padding = self.max_disp
        for d in range(self.max_disp + 1):
            tmp = np.zeros((h, w - d))
            tmp = aggregate(costl[:, d:w], costr[:, :w - d], phi_l[:, d:w])
            # tmp = single_channel_agg(tmp, costl[:, d:w], phi_l[:, d:w])
            # tmp = guidedFilter(guide=Il[:, d:w], src=tmp.astype(np.uint8), radius=1, eps=50, dDepth=-1)
            # tmp = cv2.bilateralFilter(tmp.astype(np.float32), 5, 9, 16)
            tmp_l = np.hstack((np.full((h, d), padding), tmp))
            tmp_l = np.clip(tmp_l, 0, 255)
            cost_matrix_left[d] = tmp_l
            # tmp = aggregate(costl[:, d:w], costr[:, :w-d], phi_r[:, d:w])
            # tmp = single_channel_agg(tmp, costl[:, d:w], phi_r[:, d:w])
            # tmp = guidedFilter(guide=Il[:, d:w], src=tmp.astype(np.uint8), radius=15, eps=100, dDepth=-1)
            tmp_r = np.hstack((tmp, np.full((h, d), padding)))
            tmp_r = np.clip(tmp_r, 0, 255)
            cost_matrix_right[d] = tmp_r
        # cost_matrix_left, cost_matrix_right = feature_aggregate(self.max_disp+1, h, w, cost_matrix_left, cost_matrix_right, costl, costr, phi_l, phi_r)
        if self.args.log_disp:
            show_costs(cost_matrix_left)

        return cost_matrix_left, cost_matrix_right

    def cost_aggregate_h(self, cost, U):
        y, x = np.indices((self.h, self.w))
        # calculate h integral images
        cost_h_integral = np.concatenate((np.zeros((self.max_disp + 1, self.h, 1)), cost), axis=2)
        E_h = np.zeros_like(cost)
        for w in range(1, self.w+1):
            cost_h_integral[:, :, w] += cost_h_integral[:, :, w - 1]
        for d in range(self.max_disp + 1):
            E_h[d] = cost_h_integral[d, y, x + U['h_p'][d] + 1] - \
                     cost_h_integral[d, y, x - U['h_m'][d]]
        A_h = U['h_p'] + U['h_m'] + 1

        # calculate full integral images
        cost_v_integral = np.concatenate((np.zeros((self.max_disp + 1, 1, self.w)), E_h), axis=1)
        A_v_integral = np.concatenate((np.zeros((self.max_disp + 1, 1, self.w)), A_h), axis=1)
        E_full = np.zeros_like(cost)
        A_full = np.zeros_like(cost)
        for h in range(1, self.h+1):
            cost_v_integral[:, h, :] += cost_v_integral[:, h - 1, :]
            A_v_integral[:, h, :] += A_v_integral[:, h - 1, :]
        for d in range(self.max_disp + 1):
            E_full[d] = cost_v_integral[d, y + U['v_p'][d] + 1, x] - \
                        cost_v_integral[d, y - U['v_m'][d], x]
            A_full[d] = A_v_integral[d, y + U['v_p'][d] + 1, x] - \
                        A_v_integral[d, y - U['v_m'][d], x]
        A_full[A_full == 0] = 1
        return E_full / A_full

    def cost_aggregate_v(self, cost, U):
        y, x = np.indices((self.h, self.w))
        # calculate v integral images
        cost_v_integral = np.concatenate((np.zeros((self.max_disp + 1, 1, self.w)), cost), axis=1)
        E_v = np.zeros_like(cost)
        for h in range(1, self.h+1):
            cost_v_integral[:, h, :] += cost_v_integral[:, h - 1, :]
        for d in range(self.max_disp + 1):
            E_v[d] = cost_v_integral[d, y + U['v_p'][d] + 1, x] - \
                     cost_v_integral[d, y - U['v_m'][d], x]
        A_v = U['v_p'] + U['v_m'] + 1

        # calculate full integral images
        cost_h_integral = np.concatenate((np.zeros((self.max_disp + 1, self.h, 1)), E_v), axis=2)
        A_h_integral = np.concatenate((np.zeros((self.max_disp + 1, self.h, 1)), A_v), axis=2)
        E_full = np.zeros_like(cost)
        A_full = np.zeros_like(cost)
        for w in range(1, self.w+1):
            cost_h_integral[:, :, w] += cost_h_integral[:, :, w - 1]
            A_h_integral[:, :, w] += A_h_integral[:, :, w - 1]
        for d in range(self.max_disp + 1):
            E_full[d] = cost_h_integral[d, y, x + U['h_p'][d] + 1] - \
                        cost_h_integral[d, y, x - U['h_m'][d]]
            A_full[d] = A_h_integral[d, y, x + U['h_p'][d] + 1] - \
                        A_h_integral[d, y, x - U['h_m'][d]]
        A_full[A_full == 0] = 1
        return E_full / A_full

    def cost_merge(self,c1,c2,c3):
        # c1 : census
        # c2 : base
        # c3 : bsm 
        ld_1 = 100
        ld_2 = 100
        ld_3 = 100
        costs = (1-np.exp(-c1/ld_1))+(1-np.exp(-c2/ld_2))+(1-np.exp(-c3/ld_3))
        costs /= 3
        return costs

    def census_cost_L(self,ww,wl, max_disp):
        area = (2*wl+1)*(2*ww+1)
        cost_matrix_left = np.zeros((max_disp,self.h,self.w))
        print("Left cost...")
        for d in tqdm(range(max_disp)):
            censusR_shift = np.concatenate( ( np.repeat(self.census_array_R[:,0,:].reshape(self.h,1,area), d, axis=1  ), self.census_array_R[:,:self.w-d,:] ),axis=1)
            cost_matrix_left[d,:,:] = np.sum((np.logical_xor(self.census_array_L,censusR_shift)),axis=2)
    
        return cost_matrix_left

    def census_cost_R(self,ww,wl,max_disp):
        area = (2*wl+1)*(2*ww+1)
        cost_matrix_right = np.zeros((max_disp,self.h,self.w))
        print("Right cost...")
        for d in tqdm(range(max_disp)):
            censusL_shift = np.concatenate( (self.census_array_L[:,d:,:], ( np.repeat(self.census_array_L[:,-1,:].reshape(self.h,1,area), d, axis=1 ) ) ),axis=1)
            cost_matrix_right[d,:,:] = np.sum((np.logical_xor(self.census_array_R,censusL_shift)),axis=2)
    
        return cost_matrix_right

    def build_census_window(self,ww,wl,I_l,I_r):
        area = (2*wl+1)*(2*ww+1)
        
        meanL = np.mean(I_l,axis=2)
        meanR = np.mean(I_r,axis=2)
        meanL_pad = cv2.copyMakeBorder(meanL,wl,wl,ww,ww,cv2.BORDER_REPLICATE)
        meanR_pad = cv2.copyMakeBorder(meanR,wl,wl,ww,ww,cv2.BORDER_REPLICATE)
    
        census_array_L = np.zeros((self.h,self.w,area))
        census_array_R = np.zeros((self.h,self.w,area))
        for x in tqdm(range(ww,self.w+ww)):
            for y in range(wl,self.h+wl):
                _x = x - ww
                _y = y - wl
                winL = meanL_pad[y-wl:y+wl+1,x-ww:x+ww+1].reshape(area)
                winR = meanR_pad[y-wl:y+wl+1,x-ww:x+ww+1].reshape(area)
                census_array_L[_y,_x,:] = np.array(winL >= winL[int((area+1)/2)])
                census_array_R[_y,_x,:] = np.array(winR >= winR[int((area+1)/2)])
        self.census_array_L = census_array_L
        self.census_array_R = census_array_R

    def improved_method(self, I_l, I_r):
        self.h, self.w, _ = I_l.shape
        print('Computing arms...')
        arms_l = self.compute_arms(I_l)
        arms_r = self.compute_arms(I_r)

        print('Computing U ...')
        U_l, U_r = self.get_U(arms_l, arms_r)

        # print("Creating census window...")
        # self.build_census_window(11,11,I_l,I_r)
        # print("Computing census cost...")
        # cost_census_l = self.census_cost_L(11, 11, self.max_disp+1)
        # cost_census_r = self.census_cost_R(11, 11, self.max_disp+1)

        print("Computing pixel-wise cost for each disparity...")

        census_mgr = CensusCostMgr(r=2, max_disp=self.max_disp)
        cost_census_l, cost_census_r = census_mgr.get_cost(I_l, I_r)
        # cost_base_l, cost_base_r = self.get_cost(I_l, I_r)
        cost_base_l, cost_base_r = self.base_method(I_l, I_r)
        cost_bsm_l, cost_bsm_r = self.get_cost_BSM(I_l, I_r)
        cost_volume_l = cost_census_l
        cost_volume_r = cost_census_r


        # cost_volume_l = self.cost_merge(cost_census_l, cost_base_l, cost_bsm_l)
        # cost_volume_r = self.cost_merge(cost_census_r, cost_base_r, cost_bsm_r)

        # return cost_l, cost_r

        for _ in range(1):
            print("Aggregating horizontal cost...")
            cost_volume_l = self.cost_aggregate_h(cost_volume_l, U_l)
            cost_volume_r = self.cost_aggregate_h(cost_volume_r, U_r)
            print("Aggregating vertical cost...")
            cost_volume_l = self.cost_aggregate_v(cost_volume_l, U_l)
            cost_volume_r = self.cost_aggregate_v(cost_volume_r, U_r)
        # for x in range(cost_volume_l.shape[0]):
        #     cost_volume_l[x] = guidedFilter(guide=I_l, src=cost_volume_l[x].astype(np.uint8)*3, radius=1, eps=4, dDepth=-1)
        #     cost_volume_r[x] = guidedFilter(guide=I_r, src=cost_volume_r[x].astype(np.uint8)*3, radius=1, eps=4, dDepth=-1)

        if self.args.log_disp:
            show_costs(cost_volume_l)
        return cost_volume_l, cost_volume_r


class CensusCostMgr:
    def __init__(self, r, max_disp):
        self.r = r
        self.k = 2*r + 1
        self.max_disp = max_disp
        self.census_l, self.census_r = None, None

    def compute_census(self, I):
        h, w, ch = I.shape
        I_padded = np.pad(I, ((self.r,), (self.r,), (0,)), mode='constant')
        y, x = np.indices((h, w))
        census = np.zeros((self.k ** 2, h, w, ch))
        for i in range(self.k):
            for j in range(self.k):
                I_shift = I_padded[y+i, x+j]
                census[i*self.k + j] = I < I_shift
        return census

    def get_cost(self, I_l, I_r):
        print("==> Compute census cost...")
        h, w, ch = I_l.shape
        self.census_l = self.compute_census(I_l)
        self.census_r = self.compute_census(I_r)

        cost_l = np.zeros((self.max_disp + 1, h, w))
        cost_r = np.zeros((self.max_disp + 1, h, w))
        for d in range(self.max_disp + 1):
            # census: (k, h, w, ch)
            diff = (self.census_l[:, :, d:, :] != self.census_r[:, :, :w-d, :]).sum(axis=-1).sum(axis=0)
            cost_l[d] = np.concatenate((np.full((h, d), 999), diff), axis=1)
            cost_r[d] = np.concatenate((diff, np.full((h, d), 999)), axis=1)
        return cost_l, cost_r

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Disparity Estimation')
#     parser.add_argument('--max_disp', default=60, type=int, help='maximum disparity possible')
#     parse_from_costmgr(parser)
#     args = parser.parse_args()
#
#     my_costmgr = AdaptiveCostMgr(args)
#
#     img_path = '../data/Synthetic/TL0.png'
#     test_img = cv2.imread(img_path)
#     arms = my_costmgr.compute_arms(test_img)
#     print("Done")
