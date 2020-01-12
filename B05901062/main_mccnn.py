import numpy as np
import argparse
import cv2
import time
from util import writePFM, readPFM, cal_avgerr
from solver import Solver
from super_solver import Super_Solver

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TLD0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    disp = np.zeros((h, w), dtype=np.float32)

    # TODO: Some magic

    # hyperparameter
    sgm_p1 = 1.3                # KITTI2012: 1.32   KITTI2015: 2.3    Middlebury: 1.3
    sgm_p2 = 18.1               # KITTI2012: 32     KITTI2015: 55.8   Middlebury: 18.1
    sgm_q1 = 4.5                # KITTI2012: 3      KITTI2015: 3      Middlebury: 4.5
    sgm_q2 = 9                  # KITTI2012: 6      KITTI2015: 6      Middlebury: 9
    sgm_d = 0.13                # KITTI2012: 0.08   KITTI 2015: 0.08  Middlebury: 0.13
    sgm_v = 2.75                # KITTI2012: 2      KITTI2015: 1.75   Middlebury: 2.75
    cbca_intensity = 0.02       # KITTI2012: 0.13   KITTI2015: 0.03   Middlebury: 0.02
    cbca_distance = 14          # KITTI2012: 5      KITTI2015: 5      Middlebury: 14
    cbca_num_iterations_1 = 2   # KITTI2012: 2      KITTI2015: 2      Middlebury: 2
    cbca_num_iterations_2 = 2  # KITTI2012: 0      KITTI2015: 4      Middlebury: 16
    dmax = 40

    solver = Solver(Il, Ir, dmax)

    tic = time.time()
    matching_cost_array = -solver.compute_similarity()
    toc = time.time()
    print('compute similarity: ', toc - tic)

####
    img_left_grayscale = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(int)
    img_right_grayscale = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(int)
    h, w = img_left_grayscale.shape
    
    # matching_cost_array = np.random.randn(h, w, dmax) 
    matching_cost_array = np.pad(matching_cost_array, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
#     matching_cost_array = padding_3D(matching_cost_array, cbca_distance - 1)
    # output: (h + 13 * 2, w + 13 * 2, dmax)
    
    model = Super_Solver(img_left_grayscale, img_right_grayscale, cbca_intensity, cbca_distance, cbca_num_iterations_1, dmax, sgm_p1, sgm_p2, sgm_q1, sgm_q2, sgm_d, sgm_v)
    model.img_preprocessing()
    model.compute_arm_length()
    
    tic = time.time()
    model.support_region_left()
    model.support_region_right()
    toc = time.time()
    print('compute support region: ', toc - tic)

    tic = time.time()                                                                                                                                 
    matching_cost_array_left_before_SGM = model.compute_matching_cost_left(matching_cost_array, cbca_num_iterations_1)     
    matching_cost_array_right_before_SGM = model.compute_matching_cost_right(matching_cost_array, cbca_num_iterations_1)
    # output: (h + 2 * center, w + 2 * center, dmax, iteration + 1)  with padding
#     print(matching_cost_array_left_before_SGM)
    toc = time.time()
    print('compute matching_cost_before_SGM: ', toc - tic)

    tic = time.time()
    SGM_array_left = model.SGM_left(matching_cost_array_left_before_SGM)
    SGM_array_right = model.SGM_right(matching_cost_array_right_before_SGM)
    # output: (h, w, dmax)
    toc = time.time()
    print('compute SGM: ', toc - tic)
    
    SGM_padding_array_left = np.pad(SGM_array_left, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
    SGM_padding_array_right = np.pad(SGM_array_right, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
#     SGM_padding_array_left = padding_3D(SGM_array_left, cbca_distance - 1)
#     SGM_padding_array_right = padding_3D(SGM_array_right, cbca_distance - 1)
    tic = time.time()
    matching_cost_array_left_after_SGM = model.compute_matching_cost_left(SGM_padding_array_left, cbca_num_iterations_2)
    matching_cost_array_right_after_SGM = model.compute_matching_cost_right(SGM_padding_array_right, cbca_num_iterations_2)
    # output: (h + 2 * center, w + 2 * center, dmax, iteration + 1)
    toc = time.time()
    print('compute matching_cost_after_SGM: ', toc - tic)

    # de-pad
    matching_cost_array_left_after_SGM = matching_cost_array_left_after_SGM[cbca_distance - 1 : h + cbca_distance - 1, cbca_distance - 1 : w + cbca_distance - 1, :, :]
    matching_cost_array_right_after_SGM = matching_cost_array_right_after_SGM[cbca_distance - 1 : h + cbca_distance - 1, cbca_distance - 1 : w + cbca_distance - 1, :, :]
    

    tic = time.time()
    model.compute_left_disparity_map(matching_cost_array_left_after_SGM[:, :, :, cbca_num_iterations_2])
    model.compute_right_disparity_map(matching_cost_array_right_after_SGM[:, :, :, cbca_num_iterations_2])
    # output: (h, w)
    toc = time.time()
    print('compute disparity map: ', toc - tic)

    tic = time.time()
    int_disparity_map = model.check_validity()
    # output: (h, w)
    toc = time.time()
    print('compute INT_disparity_map: ', toc - tic)

    tic = time.time()
    se_disparity_map = model.SE(int_disparity_map, SGM_array_left)
    MB = cv2.medianBlur(se_disparity_map.astype('uint8'), 5)
    blurred = cv2.bilateralFilter(MB, 5, 9, 16) 
    # cv2.imwrite(...)
    toc = time.time()
    print('finish: ', toc - tic)
####

    disp = blurred.astype(np.float32)
    return disp


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))

if __name__ == '__main__':
    main()
