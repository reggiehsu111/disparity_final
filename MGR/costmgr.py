import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def parse_from_costmgr(parser):
    return parser

def hammingDistance(n1, n2) : 
  
    x = n1 ^ n2  
    setBits = 0
  
    while (x > 0) : 
        setBits += x & 1
        x >>= 1
      
    return setBits  

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

def compute_cost(tau_l, tau_r, weight, bits):
    temp = np.bitwise_xor(tau_l.astype(np.uint8), tau_r.astype(np.uint8))
    total = np.bitwise_and(temp,weight.astype(np.uint8))
    x = bits
    setBits = np.zeros(total.shape)
    while (x > 0) : 
        setBits += total & 1
        total >>= 1
        x -= 1
    return setBits*10

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
            # h, w, ch = in_img_l.shape
            out_img_l, out_img_r  = self.get_cost(in_img_l, in_img_r)
            # Change this line of code
            # out_img_l, out_img_r = np.zeros((self.max_disp+1, h, w)), np.zeros((self.max_disp+1, h, w))
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

    def get_cost(self, Il, Ir):

        h, w, ch = Il.shape
        Il_lab = cv2.cvtColor(Il.astype(np.uint8), cv2.COLOR_RGB2LAB)
        # Array to store disparities for window sliding left, i.e. Ir sliding right
        cost_matrix_left = np.zeros((self.max_disp+1, h, w))
        # Array to store disparities for window sliding right
        cost_matrix_right = np.zeros((self.max_disp+1, h, w))
        tau_matrix = np.zeros((h, w))
        weight_matrix = np.zeros((h, w))

        Il = cv2.GaussianBlur(Il, (3,3), 3)
        Ir = cv2.GaussianBlur(Ir, (3,3), 3)
        # Find pixel pairs
        S = 10
        print("Start calculating tau and weights...")
        # for y in tqdm(range(Il.shape[0])):
        #     for x in range(Il.shape[1]):
        #         tau = 0
        #         weights = []
        #         # Sample 32 pairs for each pixel
        #         total_bits = 32
        #         # pairs = np.around(np.random.normal(0, 3, total_bits*4))
        #         # pos_ys = np.min(np.vstack((y+pairs[:total_bits//2], np.full((total_bits//2),h-1))), axis=0)
        #         # print(pos_ys)
        #         # print("x,y:", x, y)
        #         for i in range(32):
        #             pair = (np.around(np.random.normal(0, 8, 4)))
        #             # Clip the pairs
        #             pos = [int(max(0, min(y+pair[0], h-1))), int(max(0, min(x+pair[1], w-1))), int(max(0, min(y+pair[2], h-1))), int(max(0, min(x+pair[3], w-1)))]
        #             # Compute tau for BRIEF
        #             tau = (tau<<1) + binary_mask_tau(Il[pos[0], pos[1]], Il[pos[2], pos[3]])
        #             # Compute SAD for each pair and compute weights
        #             weights.append(max(np.sum(np.abs(Il_lab[y,x] - Il_lab[pos[0], pos[1]])), np.sum(np.abs(Il_lab[y,x] - Il_lab[pos[2], pos[3]]))))
        #         # print("weights:",weights)
        #         weights = np.array(weights)
        #         weights_sorted = np.sort(weights)
        #         T = weights_sorted[7]
        #         # Compute binary mask
        #         temp_weight = 0
        #         for i in weights:
        #             temp_weight = (temp_weight<<1) + binary_mask(i,T)
        #         # print("tau:", bin(tau))
        #         # print("weights:", bin(temp_weight))
        #         tau_matrix[y,x] = tau
        #         weight_matrix[y,x] = temp_weight
        # print("Finish calculating tau and weights...")
        # with open("temp_tau.pkl", "wb") as f:
            # pickle.dump(tau_matrix, f)
        # with open("temp_weight.pkl","wb") as f:
            # pickle.dump(weight_matrix, f)
        with open("temp_tau.pkl", "rb") as f:
            tau_matrix = pickle.load(f)
        with open("temp_weight.pkl", "rb") as f:
            weight_matrix = pickle.load(f)

        padding = 0
        for d in range(self.max_disp+1):
            tmp = np.zeros((h,w-d))
            tmp = compute_cost(tau_matrix[:, d:w], tau_matrix[:, :w-d], weight_matrix[:, d:w], 32)
            tmp_l = np.hstack((np.full((h, d), padding), tmp))
            tmp_l = np.clip(tmp_l, 0, 255)
            cost_matrix_left[d] = tmp_l
            tmp_r = np.hstack((tmp, np.full((h, d), padding)))
            tmp_r = np.clip(tmp_r, 0, 255)
            cost_matrix_right[d] = tmp_r



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

    def cost_aggregate(self):
        return