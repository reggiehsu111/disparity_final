import argparse
import cv2
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

WINDOW_SIZE = 37

def parse_from_costmgr(parser):
    parser.add_argument('--arms_th', default=10, type=float, help='the threshold for computing arms')
    parser.add_argument('--log_disp', action='store_true', help='Specify to log all levels of disparity map')
    return parser

def find_disp(i, w, h, max_disp, N, costl, costr, phi_l, phi_r):
    labels_l = np.zeros((w, max_disp+1))
    labels_r = np.zeros((w, max_disp+1))
    for j in range(w):
        for d in range(max_disp+1):
            if j - d < 0:
                labels_l[j, d] = N
                continue
            tmp = (costl[i, j] != costr[i, j - d]) & phi_l[i, j]
            dis = np.sum(tmp)
            labels_l[j,d] = dis
        for d in range(max_disp+1):
            if j + d >= w:
                labels_r[j, d] = N
                break
            tmp = (costl[i, j + d] != costr[i, j]) & phi_r[i, j]
            dis = np.sum(tmp)
            labels_r[j,d] = dis
    return [labels_l, labels_r]

def compute_cost(i, w, h, Il_gray, Il_lab, Ir_gray, Ir_lab, N):
    SIGMA = 6
    dx1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy1 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dx2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    dy2 = np.around(np.random.normal(0, SIGMA, N)).astype('int')
    costl = np.zeros((w, N), dtype=np.bool)
    costr = np.zeros((w, N), dtype=np.bool)
    phi_l = np.zeros((w, N), dtype=np.bool)
    phi_r = np.zeros((w, N), dtype=np.bool)
    px = np.clip(i + dx1, max(i - WINDOW_SIZE // 2, 0), min(i + WINDOW_SIZE // 2, h - 1))
    qx = np.clip(i + dx2, max(i - WINDOW_SIZE // 2, 0), min(i + WINDOW_SIZE // 2, h - 1))
    for j in range(w):
        py = np.clip(j + dy1, max(j - WINDOW_SIZE // 2, 0), min(j + WINDOW_SIZE // 2, w - 1))
        qy = np.clip(j + dy2, max(j - WINDOW_SIZE // 2, 0), min(j + WINDOW_SIZE // 2, w - 1))
        costl[j] = (Il_gray[px, py] > Il_gray[qx, qy])
        costr[j] = (Ir_gray[px, py] > Ir_gray[qx, qy])
        Wl = np.amax(
            np.array([
                np.sum(np.abs(Il_lab[i, j] - Il_lab[px, py]), axis=1),
                np.sum(np.abs(Il_lab[i, j] - Il_lab[qx, qy]), axis=1)
            ]),
            axis=0
        )
        T = np.percentile(Wl, 25)
        phi_l[j] = (Wl <= T)
        Wr = np.amax(
            np.array([
                np.sum(np.abs(Ir_lab[i, j] - Ir_lab[px, py]), axis=1),
                np.sum(np.abs(Ir_lab[i, j] - Ir_lab[qx, qy]), axis=1)
            ]),
            axis=0
        )
        T = np.percentile(Wr, 25)
        phi_r[j] = (Wr <= T)
    return [costl, costr, phi_l, phi_r]

def show_costs(matrix):
    print("Writing out...")
    for x in tqdm(range(matrix.shape[0])):
        cv2.imwrite("log/disps/"+str(x)+".jpg", matrix[x])
    return

def hammingDistance(n1, n2) : 
  
    x = n1 ^ n2  
    setBits = 0
  
    while (x > 0) : 
        setBits += x & 1
        x >>= 1
      
    return setBits 

def get_sample_pairs(S, nd):
    # Sampling nd pairs from SxS grid using gaussian
    pairs = []
    sigma = 0.04*S*S
    for i in range(nd):
        pair = (np.around(np.random.normal(0, sigma, 4)))
        # Clip the pairs
        pair = [int(max(-S, min(x, S))) for x in pair]
        pairs.append(pair)
    return pairs

def binary_mask(w,T):
    if w <= T:
        return 1
    else:
        return 0

def binary_mask_tau(pix1, pix2):
    if np.sum(pix1) > np.sum(pix2):
        return 1
    else:
        return 0




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
        padding = 0

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
        tau_matrix = np.zeros((h, w))
        weight_matrix = np.zeros((h, w))
        N = 128

        costl = []
        costr = []
        phi_l = []
        phi_r = []
        print("Getting costs...")
        for x in tqdm(range(h)):
            tmp = compute_cost(x, w, h, Il_gray, Il_lab, Ir_gray, Ir_lab, N)
            costl.append(tmp[0])
            costr.append(tmp[1])
            phi_l.append(tmp[2])
            phi_r.append(tmp[3])
        costl = np.array(costl)
        costr = np.array(costr)
        phi_l = np.array(phi_l)
        phi_r = np.array(phi_r)
        print("Aggregating...")
        labels_l = []
        labels_r = []
        for x in tqdm(range(h)):
            tmp = find_disp(x, w, h, self.max_disp, N, costl, costr, phi_l, phi_r)
            labels_l.append(tmp[0])
            labels_r.append(tmp[1])
        cost_matrix_left = np.array(labels_l)
        cost_matrix_right = np.array(labels_r)
        cost_matrix_left = np.rollaxis(cost_matrix_left, 2)
        cost_matrix_right = np.rollaxis(cost_matrix_right, 2)

        # Find pixel pairs
        # S = 10
        # nd = 32
        # print("Start calculating tau and weights...")
        # pairs = get_sample_pairs(S, nd)
        # for y in tqdm(range(Il.shape[0])):
        #     for x in range(Il.shape[1]):
        #         tau = 0
        #         weights = []
        #         for p in pairs:
        #             # Ignore points that are out of boundary
        #             if y+p[0] < 0 or y+p[0] > h-1 or  x+p[1] < 0 or x+p[1] > w-1 or y+p[2] < 0 or y+p[2] > h-1 or  x+p[3] < 0 or x+p[3] > w-1:
        #                 tau = (tau<<1)
        #                 weights.append(1000)
        #             else:
        #                 tau = (tau<<1) + binary_mask_tau(Il[y+p[0], x+p[1]], Il[y+p[2], x+p[3]])
        #                 weights.append(max(np.sum(np.abs(Il_lab[y,x] - Il_lab[y+p[0], x+p[1]])), np.sum(np.abs(Il_lab[y,x] - Il_lab[y+p[2], x+p[3]]))))


                # for i in range(32):
                #     pair = (np.around(np.random.normal(0, 8, 4)))
                #     # Clip the pairs
                #     pos = [int(max(0, min(y+pair[0], h-1))), int(max(0, min(x+pair[1], w-1))), int(max(0, min(y+pair[2], h-1))), int(max(0, min(x+pair[3], w-1)))]
                #     # Compute tau for BRIEF
                #     tau = (tau<<1) + binary_mask_tau(Il[pos[0], pos[1]], Il[pos[2], pos[3]])
                #     # Compute SAD for each pair and compute weights
                #     weights.append(max(np.sum(np.abs(Il_lab[y,x] - Il_lab[pos[0], pos[1]])), np.sum(np.abs(Il_lab[y,x] - Il_lab[pos[2], pos[3]]))))
                # print("weights:",weights)
                # weights = np.array(weights)
                # T = np.percentile(weights, 25)
                # # Compute binary mask
                # temp_weight = 0
                # for i in weights:
                #     temp_weight = (temp_weight<<1) + binary_mask(i,T)
                # # print("tau:", bin(tau))
                # # print("weights:", bin(temp_weight))
                # tau_matrix[y,x] = tau
                # weight_matrix[y,x] = temp_weight
        print("Finish calculating tau and weights...")
        # with open("temp_tau.pkl", "wb") as f:
            # pickle.dump(tau_matrix, f)
        # with open("temp_weight.pkl","wb") as f:
            # pickle.dump(weight_matrix, f)
        # with open("temp_tau.pkl", "rb") as f:
        #     tau_matrix = pickle.load(f)
        # with open("temp_weight.pkl", "rb") as f:
        #     weight_matrix = pickle.load(f)

        # padding = 0
        # for d in range(self.max_disp+1):
        #     tmp = np.zeros((h,w-d))
        #     tmp = compute_cost(tau_matrix[:, d:w], tau_matrix[:, :w-d], weight_matrix[:, d:w], nd)
        #     tmp_l = np.hstack((np.full((h, d), padding), tmp))
        #     tmp_l = np.clip(tmp_l, 0, 255)
        #     cost_matrix_left[d] = tmp_l*10
        #     tmp_r = np.hstack((tmp, np.full((h, d), padding)))
        #     tmp_r = np.clip(tmp_r, 0, 255)*10
        #     cost_matrix_right[d] = tmp_r
        if self.args.log_disp:
            show_costs(cost_matrix_left)



        # brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        # kps = []
        # # Create keypoint list:
        # for y in range(Il.shape[0]):
        #     for x in range(Il.shape[1]):
        #         kps.append(cv2.KeyPoint(x, y, 0))
        # print("Length of keypoint list:", len(kps))
        # # find the keypoints with fast
        # # compute the descriptors with BRIEF
        # kp, des_l = brief.compute(Il.astype(np.uint8), kps)
        # _, des_r = brief.compute(Ir.astype(np.uint8), kps)
        # print("kp:", len(kp))
        # des_l = np.array(des_l)
        # des_r = np.array(des_r)
        # print(des_r.shape)

        # print(des_l.shape)

        # ans = 0
        # for x in range(len(des_l[0])):
        #     print(des_l[0][x], ":", bin(des_l[0][x]))
        #     ans += (des_l[0][x] << x*8)
        # print(bin(ans))
        # for d in range(self.max_disp+1):
        #     tmp = np.zeros(h,w-d)
        #     for y in range(Il.shape[0]):
        #         for x in range(Il.shape[1]):
        #             pos = x+y*Il.shape[1]
        #             hd = 0
        #             for i in range(32):
        #                 hd += hammingDistance(des_l[pos][i], des_r[pos-d][x])
        #             cost_matrix_left[d,y,x] = hd
        #             cost_matrix_right[d,]


        
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
        cost_l = np.zeros((self.max_disp + 1, self.h, self.w))
        cost_r = np.zeros((self.max_disp + 1, self.h, self.w))
        for d in range(self.max_disp + 1):
            diff = np.abs(I_r[:, :self.w - d, :] - I_l[:, d:, :]).sum(axis=2)
            diff[diff > 255] = 255
            cost_l[d] = np.concatenate((np.full((self.h, d), 999), diff), axis=1)
            cost_r[d] = np.concatenate((diff, np.full((self.h, d), 999)), axis=1)
        return cost_l, cost_r

    def cost_aggregate_h(self, cost, U):
        y, x = np.indices((self.h, self.w))
        # calculate h integral images
        cost_h_integral = np.concatenate((np.zeros((self.max_disp + 1, self.h, 1)), cost), axis=2)
        # cost_h_integral = np.concatenate((cost, np.zeros((self.max_disp + 1, self.h, 1))), axis=2)
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

    def improved_method(self, I_l, I_r):
        self.h, self.w, _ = I_l.shape
        print('Computing arms...')
        arms_l = self.compute_arms(I_l)
        arms_r = self.compute_arms(I_r)

        print('Computing U ...')
        U_l, U_r = self.get_U(arms_l, arms_r)
        print("Computing pixel-wise cost for each disparity...")
        cost_volume_l, cost_volume_r = self.get_cost(I_l, I_r)
        # return cost_l, cost_r

        for _ in range(1):
            print("Aggregating horizontal cost...")
            cost_volume_l = self.cost_aggregate_h(cost_volume_l, U_l)
            cost_volume_r = self.cost_aggregate_h(cost_volume_r, U_r)
            print("Aggregating vertical cost...")
            cost_volume_l = self.cost_aggregate_v(cost_volume_l, U_l)
            cost_volume_r = self.cost_aggregate_v(cost_volume_r, U_r)
        # for x in range(cost_volume_l.shape[0]):
        #     tmp = cost_volume_l[x, :, x:self.w];
        #     tmp = guidedFilter(guide=I_l[:, x:self.w], src=tmp.astype(np.uint8), radius=80, eps=200, dDepth=-1)
        #     tmp_l = np.hstack((cost_volume_l[x, :, :x], tmp))
        #     tmp_l = np.clip(tmp_l, 0, 255)
        #     cost_volume_l[x] = tmp_l
        #     tmp_r = np.hstack((tmp, cost_volume_l[x, :, :x]))
        #     tmp_r = np.clip(tmp_r, 0, 255)
        #     cost_volume_r[x] = tmp_r
        if self.args.log_disp:
            show_costs(cost_volume_l)
        return cost_volume_l, cost_volume_r

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
