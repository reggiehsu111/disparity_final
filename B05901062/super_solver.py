#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Each image was preprocessed by subtracting the mean and dividing by the standard deviation of its pixel intensity values. 
# The left and right image of a stereo pair were preprocessed separately
# Our initial experiments suggested that using color information does not improve the quality of the disparity maps; 
# therefore, we converted all color images to grayscale.
import cv2
import numpy as np
import time
# from model import MC_CNN

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
dmax = 15

class Super_Solver():
    def __init__(self, img_left, img_right, cbca_intensity, cbca_distance, cbca_num_iterations_1, dmax, sgm_p1, sgm_p2, sgm_q1, sgm_q2, sgm_d, sgm_v):
        self.h, self.w = img_left.shape
        self.img_left = img_left
        self.img_right = img_right
        self.cbca_intensity = cbca_intensity
        self.cbca_distance = cbca_distance
        self.center = cbca_distance - 1
        self.dmax = dmax
        self.sgm_p1 = sgm_p1
        self.sgm_p2 = sgm_p2
        self.sgm_q1 = sgm_q1
        self.sgm_q2 = sgm_q2
        self.sgm_d = sgm_d
        self.sgm_v = sgm_v
        self.cbca_num_iterations_1 = cbca_num_iterations_1

    def img_preprocessing(self):
        mean_left, mean_right = np.sum(self.img_left) / (self.h * self.w), np.sum(self.img_right) / (self.h * self.w)
        std_left, std_right = np.std(self.img_left), np.std(self.img_right) 
        self.preprocessed_img_left, self.preprocessed_img_right = np.absolute( self.img_left - mean_left) / std_left, np.absolute( self.img_right - mean_right) / std_right
#         return img

    def compute_arm_length(self):
#         h, w = img_left.shape
        self.arm_length_img_l_array, self.arm_length_img_r_array = np.zeros((self.h, self.w, 4), dtype = 'int8'), np.zeros((self.h, self.w, 4), dtype = 'int8') # top:0 down:1 left:2 right:3
        for x in range(self.w):
            for y in range(self.h):
                left_arm_length_img_l, right_arm_length_img_l, top_arm_length_img_l, bottom_arm_length_img_l = 0, 0, 0, 0
                left_arm_length_img_r, right_arm_length_img_r, top_arm_length_img_r, bottom_arm_length_img_r = 0, 0, 0, 0
                while x - left_arm_length_img_l - 1 > -1 and np.absolute(self.img_left[y][x - left_arm_length_img_l - 1] - self.img_left[y][x]) < self.cbca_intensity and left_arm_length_img_l < self.cbca_distance - 1:
                    left_arm_length_img_l += 1
                while x + right_arm_length_img_l + 1 < self.w and np.absolute(self.img_left[y][x + right_arm_length_img_l + 1] - self.img_left[y][x]) < self.cbca_intensity and right_arm_length_img_l < self.cbca_distance - 1:
                    right_arm_length_img_l += 1
                while y - top_arm_length_img_l - 1 > -1 and np.absolute(self.img_left[y - top_arm_length_img_l - 1][x] - self.img_left[y][x]) < self.cbca_intensity and top_arm_length_img_l < self.cbca_distance - 1:
                    top_arm_length_img_l += 1
                while y + bottom_arm_length_img_l + 1 < self.h and np.absolute(self.img_left[y + bottom_arm_length_img_l + 1][x] - self.img_left[y][x]) < self.cbca_intensity and bottom_arm_length_img_l < self.cbca_distance - 1:
                    bottom_arm_length_img_l += 1
                while x - left_arm_length_img_r - 1 > -1 and np.absolute(self.img_right[y][x - left_arm_length_img_r - 1] - self.img_right[y][x]) < self.cbca_intensity and left_arm_length_img_r < self.cbca_distance - 1: # and left_arm_length_img_r <= left_arm_length_img_l:
                    left_arm_length_img_r += 1
                while x + right_arm_length_img_r + 1 < self.w and np.absolute(self.img_right[y][x + right_arm_length_img_r + 1] - self.img_right[y][x]) < self.cbca_intensity and right_arm_length_img_r < self.cbca_distance - 1: # and right_arm_length_img_r <= right_arm_length_img_l:
                    right_arm_length_img_r += 1
                while y - top_arm_length_img_r - 1 > -1 and np.absolute(self.img_right[y - top_arm_length_img_r - 1][x] - self.img_right[y][x]) < self.cbca_intensity and top_arm_length_img_r < self.cbca_distance - 1: # and top_arm_length_img_r <= top_arm_length_img_l:
                    top_arm_length_img_r += 1
                while y + bottom_arm_length_img_r + 1 < self.h and np.absolute(self.img_right[y + bottom_arm_length_img_r + 1][x] - self.img_right[y][x]) < self.cbca_intensity and bottom_arm_length_img_r < self.cbca_distance - 1: # and bottom_arm_length_img_r <= bottom_arm_length_img_l:
                    bottom_arm_length_img_r += 1
                self.arm_length_img_l_array[y][x][0], self.arm_length_img_l_array[y][x][1], self.arm_length_img_l_array[y][x][2], self.arm_length_img_l_array[y][x][3] = top_arm_length_img_l, bottom_arm_length_img_l, left_arm_length_img_l, right_arm_length_img_l
                self.arm_length_img_r_array[y][x][0], self.arm_length_img_r_array[y][x][1], self.arm_length_img_r_array[y][x][2], self.arm_length_img_r_array[y][x][3] = top_arm_length_img_r, bottom_arm_length_img_r, left_arm_length_img_r, right_arm_length_img_r
#         return self.arm_length_img_l_array, self.arm_length_img_r_array

    def support_region_left(self):
#         h, w, direction = arm_length_img_l_array.shape
        u_i_side_length = 2 * self.cbca_distance - 1
        self.u_array_left = np.zeros( (self.h, self.w, self.dmax, u_i_side_length, u_i_side_length), dtype = 'int8' )
        for y in range(self.h):                                                
            for x in range(self.w):                                                               
                for disparity in range(self.dmax):
                    if x - disparity >= 0:
                        u_i = np.zeros( (u_i_side_length, u_i_side_length), dtype = 'int8')
                        u_top, u_bottom = np.minimum(self.arm_length_img_l_array[y][x][0], self.arm_length_img_r_array[y][x - disparity][0]), np.minimum(self.arm_length_img_l_array[y][x][1], self.arm_length_img_r_array[y][x - disparity][1])
                        u_height = u_top + u_bottom + 1
                        for i in range(u_height):
                            u_left = np.minimum(self.arm_length_img_l_array[y - u_top + i][x][2], self.arm_length_img_r_array[y - u_top + i][x - disparity][2])
                            u_right = np.minimum(self.arm_length_img_l_array[y - u_top + i][x][3], self.arm_length_img_r_array[y - u_top + i][x - disparity][3])
                            u_i[self.center - u_top + i, self.center - u_left : self.center + u_right + 1] = 1
                        u_i_size = np.sum(u_i)
                        self.u_array_left[y][x][disparity] = u_i
#         return self.u_array_left

    def support_region_right(self):
#         h, w, direction = arm_length_img_l_array.shape
        u_i_side_length = 2 * self.cbca_distance - 1
        self.u_array_right = np.zeros( (self.h, self.w, self.dmax, u_i_side_length, u_i_side_length), dtype = 'int8' )  
        for y in range(self.h):                                                
            for x in range(self.w):                                                               
                for disparity in range(self.dmax):
                    if x + disparity < self.w:
                        u_i = np.zeros( (u_i_side_length, u_i_side_length), dtype = 'int8')
                        u_top, u_bottom = np.minimum(self.arm_length_img_l_array[y][x + disparity][0], self.arm_length_img_r_array[y][x][0]), np.minimum(self.arm_length_img_l_array[y][x + disparity][1], self.arm_length_img_r_array[y][x][1])
                        u_height = u_top + u_bottom + 1
                        for i in range(u_height):
                            u_left = np.minimum(self.arm_length_img_l_array[y - u_top + i][x + disparity][2], self.arm_length_img_r_array[y - u_top + i][x][2])
                            u_right = np.minimum(self.arm_length_img_l_array[y - u_top + i][x + disparity][3], self.arm_length_img_r_array[y - u_top + i][x][3])
                            u_i[self.center - u_top + i, self.center - u_left : self.center + u_right + 1] = 1
                        u_i_size = np.sum(u_i)
                        self.u_array_right[y][x][disparity] = u_i
#         return self.u_array_right

    def compute_matching_cost_left(self, matching_cost_array, iteration): 
#         h, w, dmax, u_i_side_length, u_i_side_length = u_array_left.shape
#         center = self.cbca_distance - 1
        matching_cost_array_left = np.zeros( (self.h + 2 * self.center, self.w + 2 * self.center, self.dmax, iteration + 1) )
        matching_cost_array_left[:, :, :,0] = matching_cost_array
        for i in range(iteration):
            for y in range(self.h):
                for x in range(self.w):
                    for disparity in range(self.dmax):
                        if np.sum(self.u_array_left[y][x][disparity]) == 0:
                            matching_cost_array_left[y + self.center][x + self.center][disparity][i + 1] = 0
                    # # for disparity in range(self.dmax):
                    # if np.sum(self.u_array_left[y][x][self.dmax]) == 0:
                    #         matching_cost_array_left[y + self.center][x + self.center][disparity][i + 1] = np.logical_or(np.sum(self.u_array_left[y, x, self.dmax], axis=(1, 2)), )
                        else:
                            matching_cost_array_left[y + self.center][x + self.center][disparity][i + 1] = np.sum( np.multiply(self.u_array_left[y][x][disparity], matching_cost_array_left[y : y + 2 * self.center + 1, x : x + 2 * self.center + 1, disparity, i]) ) / np.sum(self.u_array_left[y][x][disparity])
        return matching_cost_array_left

    def compute_matching_cost_right(self, matching_cost_array, iteration): 
#         h, w, dmax, u_i_side_length, u_i_side_length = u_array_right.shape
#         center = cbca_distance - 1
        matching_cost_array_right = np.zeros( (self.h + 2 * self.center, self.w + 2 * self.center, self.dmax, iteration + 1) )
        matching_cost_array_right[:, :, :,0] = matching_cost_array
        for i in range(iteration):
            for y in range(self.h):
                for x in range(self.w):
                    for disparity in range(self.dmax):
                        if np.sum(self.u_array_right[y][x][disparity]) == 0:
                            matching_cost_array_right[y + self.center][x + self.center][disparity][i + 1] = 0
                        else:
                            matching_cost_array_right[y + self.center][x + self.center][disparity][i + 1] = np.sum( np.multiply(self.u_array_right[y][x][disparity], matching_cost_array_right[y : y + 2 * self.center + 1, x : x + 2 * self.center + 1, disparity, i]) ) / np.sum(self.u_array_right[y][x][disparity])
        return matching_cost_array_right

    def intensity_diff(self, img, x_coord, y_coord, disparity, dx, dy):
        D = np.absolute( img[y_coord + dy][x_coord - disparity + dx] - img[y_coord][x_coord - disparity] )
        return D
    
    def SGM_left(self, matching_cost_array_left_before_SGM):
#         h, w = preprocessed_img_left.shape
        cost_in_four_direction_left = np.zeros( (self.h, self.w, self.dmax, 4) ) # 0:up 1:down 2:left 3:right
        dx = [0, 0, 1, -1]
        dy = [-1, 1, 0, 0]
        for y in range(self.h):
            for x in range(self.w):
                for disparity in range(self.dmax):
                    d1, d2, p1, p2 = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                    second_item, third_one, third_two, third_three, third_item, buff = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                    # up
                    if y + dy[0] >= 0 and x - disparity >= 0:
                        d1[0] = self.intensity_diff(self.preprocessed_img_left, x, y, 0, dx[0], dy[0])
                        d2[0] = self.intensity_diff(self.preprocessed_img_right, x, y, disparity, dx[0], dy[0])
                        if d1[0] < self.sgm_d and d2[0] < self.sgm_d:
                            p1[0], p2[0] = self.sgm_p1 / self.sgm_v , self.sgm_p2
                        elif d1[0] >= self.sgm_d and d2[0] >= self.sgm_d:
                            p1[0], p2[0] = self.sgm_p1 / (self.sgm_q2 * self.sgm_v), self.sgm_p2 / self.sgm_q2
                        else:
                            p1[0], p2[0] = self.sgm_p1 / (self.sgm_q1 * self.sgm_v), self.sgm_p2 / self.sgm_q1

                        second_item[0] = min(cost_in_four_direction_left[y + dy[0], x + dx[0], : , 0])
                        third_one[0] = cost_in_four_direction_left[y + dy[0], x + dx[0], disparity, 0]
                        if disparity - 1 >= 0:
                            third_two[0] = cost_in_four_direction_left[y + dy[0], x + dx[0], disparity - 1, 0]
                        else:
                            third_two[0] = 0
                        if disparity + 1 < self.dmax:
                            third_three[0] = cost_in_four_direction_left[y + dy[0], x + dx[0], disparity + 1, 0]
                        else:
                            third_three[0] = 0
                        if disparity - 1 == 0:
                            buff[0] = min(cost_in_four_direction_left[y + dy[0], x + dx[0], disparity + 1: , 0])
                        elif disparity + 1 == self.dmax:
                            buff[0] = min(cost_in_four_direction_left[y + dy[0], x + dx[0], :disparity - 1 , 0])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_left[y + dy[0], x + dx[0], : disparity, 0], cost_in_four_direction_left[y + dy[0], x + dx[0], disparity + 1: , 0]) )
                            buff[0] = min(buffer)

                        third_item[0] = min(third_one[0], third_two[0] + p1[0], third_three[0] + p1[0], buff[0] + p2[0]) 
                        cost_in_four_direction_left[y][x][disparity][0] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[0] + third_item[0]
                    else:
                        cost_in_four_direction_left[y][x][disparity][0] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # down
                    if y + dy[1] < self.h and x - disparity >= 0:
                        d1[1] = self.intensity_diff(self.preprocessed_img_left, x, y, 0, dx[1], dy[1])
                        d2[1] = self.intensity_diff(self.preprocessed_img_right, x, y, disparity, dx[1], dy[1])
                        if d1[1] < self.sgm_d and d2[1] < self.sgm_d:
                            p1[1], p2[1] = self.sgm_p1 / self.sgm_v , self.sgm_p2
                        elif d1[1] >= self.sgm_d and d2[1] >= self.sgm_d:
                            p1[1], p2[1] = self.sgm_p1 / (self.sgm_q2 * self.sgm_v), self.sgm_p2 / self.sgm_q2
                        else:
                            p1[1], p2[1] = self.sgm_p1 / (self.sgm_q1 * self.sgm_v), self.sgm_p2 / self.sgm_q1 

                        second_item[1] = min(cost_in_four_direction_left[y + dy[1], x + dx[1], : , 1])
                        third_one[1] = cost_in_four_direction_left[y + dy[1], x + dx[1], disparity, 1]
                        if disparity - 1 >= 0:
                            third_two[1] = cost_in_four_direction_left[y + dy[1], x + dx[1], disparity - 1, 1]
                        else:
                            third_two[1] = 0
                        if disparity + 1 < self.dmax:
                            third_three[1] = cost_in_four_direction_left[y + dy[1], x + dx[1], disparity + 1, 1]
                        else:
                            third_three[1] = 0
                        if disparity - 1 == 0:
                            buff[1] = min(cost_in_four_direction_left[y + dy[1], x + dx[1], disparity + 1: , 1])
                        elif disparity + 1 == self.dmax:
                            buff[1] = min(cost_in_four_direction_left[y + dy[1], x + dx[1], : disparity - 1, 1])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_left[y + dy[1], x + dx[1], : disparity, 1], cost_in_four_direction_left[y + dy[1], x + dx[1], disparity + 1: , 1]) )
                            buff[1] = min(buffer)

                        third_item[1] = min(third_one[1], third_two[1] + p1[1], third_three[1] + p1[1], buff[1] + p2[1]) 
                        cost_in_four_direction_left[y][x][disparity][1] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[1] + third_item[1]
                    else:
                        cost_in_four_direction_left[y][x][disparity][1] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # left
                    if x + dx[2] < self.w and x - disparity >= 0:
                        d1[2] = self.intensity_diff(self.preprocessed_img_left, x, y, 0, dx[2], dy[2])
                        d2[2] = self.intensity_diff(self.preprocessed_img_right, x, y, disparity, dx[2], dy[2])
                        if d1[2] < self.sgm_d and d2[2] < self.sgm_d:
                            p1[2], p2[2] = sgm_p1 , sgm_p2
                        elif d1[2] >= self.sgm_d and d2[2] >= self.sgm_d:
                            p1[2], p2[2] = self.sgm_p1 / self.sgm_q2, self.sgm_p2 / self.sgm_q2
                        else:
                            p1[2], p2[2] = self.sgm_p1 / self.sgm_q1, self.sgm_p2 / self.sgm_q1

                        second_item[2] = min(cost_in_four_direction_left[y + dy[2], x + dx[2], : , 2])
                        third_one[2] = cost_in_four_direction_left[y + dy[2], x + dx[2], disparity, 2]
                        if disparity - 1 >= 0:
                            third_two[2] = cost_in_four_direction_left[y + dy[2], x + dx[2], disparity - 1, 2]
                        else:
                            third_two[2] = 0
                        if disparity + 1 < self.dmax:
                            third_three[2] = cost_in_four_direction_left[y + dy[2], x + dx[2], disparity + 1, 2]
                        else:
                            third_three[2] = 0
                        if disparity - 1 == 0:
                            buff[2] = min(cost_in_four_direction_left[y + dy[2], x + dx[2], disparity + 1: , 2])
                        elif disparity + 1 == self.dmax:
                            buff[2] = min(cost_in_four_direction_left[y + dy[2], x + dx[2], :disparity - 1 , 2])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_left[y + dy[2], x + dx[2], : disparity, 2], cost_in_four_direction_left[y + dy[2], x + dx[2], disparity + 1: , 2]) )
                            buff[2] = min(buffer)

                        third_item[2] = min(third_one[2], third_two[2] + p1[2], third_three[2] + p1[2], buff[2] + p2[2]) 
                        cost_in_four_direction_left[y][x][disparity][2] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[2] + third_item[2] 
                    else:
                        cost_in_four_direction_left[y][x][disparity][2] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # right
    #                 if x + dx[3] >= 0 and x - disparity + dx[3] >= 0:
                    if x + dx[3] >= disparity:
                        d1[3] = self.intensity_diff(self.preprocessed_img_left, x, y, 0, dx[3], dy[3])
                        d2[3] = self.intensity_diff(self.preprocessed_img_right, x, y, disparity, dx[3], dy[3])
                        if d1[3] < self.sgm_d and d2[3] < self.sgm_d:
                            p1[3], p2[3] = self.sgm_p1 , self.sgm_p2
                        elif d1[3] >= self.sgm_d and d2[3] >= self.sgm_d:
                            p1[3], p2[3] = self.sgm_p1 / self.sgm_q2, self.sgm_p2 / self.sgm_q2
                        else:
                            p1[3], p2[3] = self.sgm_p1 / self.sgm_q1, self.sgm_p2 / self.sgm_q1

                        second_item[3] = min(cost_in_four_direction_left[y + dy[3], x + dx[3], : , 3])
                        third_one[3] = cost_in_four_direction_left[y + dy[3], x + dx[3], disparity, 3]
                        if disparity - 1 >= 0:
                            third_two[3] = cost_in_four_direction_left[y + dy[3], x + dx[3], disparity - 1, 3]
                        else:
                            third_two[3] = 0
                        if disparity + 1 < self.dmax:
                            third_three[3] = cost_in_four_direction_left[y + dy[3], x + dx[3], disparity + 1, 3]
                        else:
                            third_three[3] = 0
                        if disparity - 1 == 0:
                            buff[3] = min(cost_in_four_direction_left[y + dy[3], x + dx[3], disparity + 1: , 3])
                        elif disparity + 1 == self.dmax:
                            buff[3] = min(cost_in_four_direction_left[y + dy[3], x + dx[3], : disparity - 1, 3])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_left[y + dy[3], x + dx[3], : disparity, 3], cost_in_four_direction_left[y + dy[3], x + dx[3], disparity + 1: , 3]) )
                            buff[3] = min(buffer)

                        third_item[3] = min(third_one[3], third_two[3] + p1[3], third_three[3] + p1[3], buff[3] + p2[3]) 
                        cost_in_four_direction_left[y][x][disparity][3] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[3] + third_item[3]
                    else:
                        cost_in_four_direction_left[y][x][disparity][3] = matching_cost_array_left_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]

        SGM_array_left = np.sum(cost_in_four_direction_left, axis = 3)
        return SGM_array_left

    def SGM_right(self, matching_cost_array_right_before_SGM):
    #     h, w = preprocessed_img_left.shape
        cost_in_four_direction_right = np.zeros( (self.h, self.w, self.dmax, 4) ) # 0:up 1:down 2:left 3:right
        dx = [0, 0, 1, -1]
        dy = [-1, 1, 0, 0]
        for y in range(self.h):
            for x in range(self.w):
                for disparity in range(self.dmax):
                    d1, d2, p1, p2 = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                    second_item, third_one, third_two, third_three, third_item, buff = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                    # up
                    if y + dy[0] >= 0 and x + disparity < self.w:
                        d1[0] = self.intensity_diff(self.preprocessed_img_left, x, y, - disparity, dx[0], dy[0])
                        d2[0] = self.intensity_diff(self.preprocessed_img_right, x, y, 0, dx[0], dy[0])
                        if d1[0] < self.sgm_d and d2[0] < self.sgm_d:
                            p1[0], p2[0] = self.sgm_p1 / self.sgm_v , self.sgm_p2
                        elif d1[0] >= self.sgm_d and d2[0] >= self.sgm_d:
                            p1[0], p2[0] = self.sgm_p1 / (self.sgm_q2 * self.sgm_v), self.sgm_p2 / self.sgm_q2
                        else:
                            p1[0], p2[0] = self.sgm_p1 / (self.sgm_q1 * self.sgm_v), self.sgm_p2 / self.sgm_q1

                        second_item[0] = min(cost_in_four_direction_right[y + dy[0], x + dx[0], : , 0])
                        third_one[0] = cost_in_four_direction_right[y + dy[0], x + dx[0], disparity, 0]
                        if disparity - 1 >= 0:
                            third_two[0] = cost_in_four_direction_right[y + dy[0], x + dx[0], disparity - 1, 0]
                        else:
                            third_two[0] = 0
                        if disparity + 1 < self.dmax:
                            third_three[0] = cost_in_four_direction_right[y + dy[0], x + dx[0], disparity + 1, 0]
                        else:
                            third_three[0] = 0
                        if disparity - 1 == 0:
                            buff[0] = min(cost_in_four_direction_right[y + dy[0], x + dx[0], disparity + 1: , 0])
                        elif disparity + 1 == self.dmax:
                            buff[0] = min(cost_in_four_direction_right[y + dy[0], x + dx[0], :disparity - 1 , 0])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_right[y + dy[0], x + dx[0], : disparity, 0], cost_in_four_direction_right[y + dy[0], x + dx[0], disparity + 1: , 0]) )
                            buff[0] = min(buffer)

                        third_item[0] = min(third_one[0], third_two[0] + p1[0], third_three[0] + p1[0], buff[0] + p2[0]) 
                        cost_in_four_direction_right[y][x][disparity][0] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[0] + third_item[0]
                    else:
                        cost_in_four_direction_right[y][x][disparity][0] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # down
                    if y + dy[1] < self.h and x + disparity < self.w:
                        d1[1] = self.intensity_diff(self.preprocessed_img_left, x, y, - disparity, dx[1], dy[1])
                        d2[1] = self.intensity_diff(self.preprocessed_img_right, x, y, 0, dx[1], dy[1])
                        if d1[1] < self.sgm_d and d2[1] < self.sgm_d:
                            p1[1], p2[1] = self.sgm_p1 / self.sgm_v , self.sgm_p2
                        elif d1[1] >= self.sgm_d and d2[1] >= self.sgm_d:
                            p1[1], p2[1] = self.sgm_p1 / (self.sgm_q2 * self.sgm_v), self.sgm_p2 / self.sgm_q2
                        else:
                            p1[1], p2[1] = self.sgm_p1 / (self.sgm_q1 * self.sgm_v), self.sgm_p2 / self.sgm_q1 

                        second_item[1] = min(cost_in_four_direction_right[y + dy[1], x + dx[1], : , 1])
                        third_one[1] = cost_in_four_direction_right[y + dy[1], x + dx[1], disparity, 1]
                        if disparity - 1 >= 0:
                            third_two[1] = cost_in_four_direction_right[y + dy[1], x + dx[1], disparity - 1, 1]
                        else:
                            third_two[1] = 0
                        if disparity + 1 < self.dmax:
                            third_three[1] = cost_in_four_direction_right[y + dy[1], x + dx[1], disparity + 1, 1]
                        else:
                            third_three[1] = 0
                        if disparity - 1 == 0:
                            buff[1] = min(cost_in_four_direction_right[y + dy[1], x + dx[1], disparity + 1: , 1])
                        elif disparity + 1 == self.dmax:
                            buff[1] = min(cost_in_four_direction_right[y + dy[1], x + dx[1], : disparity - 1, 1])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_right[y + dy[1], x + dx[1], : disparity, 1], cost_in_four_direction_right[y + dy[1], x + dx[1], disparity + 1: , 1]) )
                            buff[1] = min(buffer)

                        third_item[1] = min(third_one[1], third_two[1] + p1[1], third_three[1] + p1[1], buff[1] + p2[1]) 
                        cost_in_four_direction_right[y][x][disparity][1] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[1] + third_item[1]
                    else:
                        cost_in_four_direction_right[y][x][disparity][1] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # left
    #                 if x + dx[2] < w and x + disparity + dx[2] < w:
                    if x + disparity + dx[2] < self.w:
                        d1[2] = self.intensity_diff(self.preprocessed_img_left, x, y, - disparity, dx[2], dy[2])
                        d2[2] = self.intensity_diff(self.preprocessed_img_right, x, y, 0, dx[2], dy[2])
                        if d1[2] < self.sgm_d and d2[2] < self.sgm_d:
                            p1[2], p2[2] = self.sgm_p1 , self.sgm_p2
                        elif d1[2] >= self.sgm_d and d2[2] >= self.sgm_d:
                            p1[2], p2[2] = self.sgm_p1 / self.sgm_q2, self.sgm_p2 / self.sgm_q2
                        else:
                            p1[2], p2[2] = self.sgm_p1 / self.sgm_q1, self.sgm_p2 / self.sgm_q1

                        second_item[2] = min(cost_in_four_direction_right[y + dy[2], x + dx[2], : , 2])
                        third_one[2] = cost_in_four_direction_right[y + dy[2], x + dx[2], disparity, 2]
                        if disparity - 1 >= 0:
                            third_two[2] = cost_in_four_direction_right[y + dy[2], x + dx[2], disparity - 1, 2]
                        else:
                            third_two[2] = 0
                        if disparity + 1 < self.dmax:
                            third_three[2] = cost_in_four_direction_right[y + dy[2], x + dx[2], disparity + 1, 2]
                        else:
                            third_three[2] = 0
                        if disparity - 1 == 0:
                            buff[2] = min(cost_in_four_direction_right[y + dy[2], x + dx[2], disparity + 1: , 2])
                        elif disparity + 1 == self.dmax:
                            buff[2] = min(cost_in_four_direction_right[y + dy[2], x + dx[2], :disparity - 1 , 2])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_right[y + dy[2], x + dx[2], : disparity, 2], cost_in_four_direction_right[y + dy[2], x + dx[2], disparity + 1: , 2]) )
                            buff[2] = min(buffer)

                        third_item[2] = min(third_one[2], third_two[2] + p1[2], third_three[2] + p1[2], buff[2] + p2[2]) 
                        cost_in_four_direction_right[y][x][disparity][2] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[2] + third_item[2] 
                    else:
                        cost_in_four_direction_right[y][x][disparity][2] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]
                    # right
                    if x + dx[3] >= 0 and x + disparity < self.w:
                        d1[3] = self.intensity_diff(self.preprocessed_img_left, x, y, - disparity, dx[3], dy[3])
                        d2[3] = self.intensity_diff(self.preprocessed_img_right, x, y, 0, dx[3], dy[3])
                        if d1[3] < self.sgm_d and d2[3] < self.sgm_d:
                            p1[3], p2[3] = self.sgm_p1 , self.sgm_p2
                        elif d1[3] >= self.sgm_d and d2[3] >= self.sgm_d:
                            p1[3], p2[3] = self.sgm_p1 / self.sgm_q2, self.sgm_p2 / self.sgm_q2
                        else:
                            p1[3], p2[3] = self.sgm_p1 / self.sgm_q1, self.sgm_p2 / self.sgm_q1

                        second_item[3] = min(cost_in_four_direction_right[y + dy[3], x + dx[3], : , 3])
                        third_one[3] = cost_in_four_direction_right[y + dy[3], x + dx[3], disparity, 3]
                        if disparity - 1 >= 0:
                            third_two[3] = cost_in_four_direction_right[y + dy[3], x + dx[3], disparity - 1, 3]
                        else:
                            third_two[3] = 0
                        if disparity + 1 < self.dmax:
                            third_three[3] = cost_in_four_direction_right[y + dy[3], x + dx[3], disparity + 1, 3]
                        else:
                            third_three[3] = 0
                        if disparity - 1 == 0:
                            buff[3] = min(cost_in_four_direction_right[y + dy[3], x + dx[3], disparity + 1: , 3])
                        elif disparity + 1 == self.dmax:
                            buff[3] = min(cost_in_four_direction_right[y + dy[3], x + dx[3], : disparity - 1, 3])
                        else:
                            buffer = np.concatenate( (cost_in_four_direction_right[y + dy[3], x + dx[3], : disparity, 3], cost_in_four_direction_right[y + dy[3], x + dx[3], disparity + 1: , 3]) )
                            buff[3] = min(buffer)

                        third_item[3] = min(third_one[3], third_two[3] + p1[3], third_three[3] + p1[3], buff[3] + p2[3]) 
                        cost_in_four_direction_right[y][x][disparity][3] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1] - second_item[3] + third_item[3]
                    else:
                        cost_in_four_direction_right[y][x][disparity][3] = matching_cost_array_right_before_SGM[y + self.cbca_distance - 1][x + self.cbca_distance - 1][disparity][self.cbca_num_iterations_1]

        SGM_array_right = np.sum(cost_in_four_direction_right, axis = 3) * 0.25
        return SGM_array_right

    def compute_left_disparity_map(self, matching_cost_array_left_after_SGM):
#         h, w, disparity = matching_cost_array_left_after_SGM.shape
        self.left_disparity_map = np.zeros((self.h, self.w), dtype = 'int8')
        for y in range(self.h):
            for x in range(self.w):
                self.left_disparity_map[y][x] = np.argmax(matching_cost_array_left_after_SGM[y][x])
        # return self.left_disparity_map

    def compute_right_disparity_map(self, matching_cost_array_right_after_SGM):
#         h, w, disparity = matching_cost_array_right_after_SGM.shape
        self.right_disparity_map = np.zeros((self.h, self.w), dtype = 'int8')
        for y in range(self.h):
            for x in range(self.w):
                self.right_disparity_map[y][x] = np.argmax(matching_cost_array_right_after_SGM[y][x])
        # return self.right_disparity_map

    def check_validity(self):
#         h, w = left_disparity_map.shape
        mismatch_range = int(self.dmax / 8)
        validity_map = np.zeros((self.h, self.w), dtype = 'int8')  
        for x in range(self.w):
            for y in range(self.h):
                if abs( self.left_disparity_map[y][x] - self.right_disparity_map[y][x - self.left_disparity_map[y][x]] ) <= 1:  # correct
                    validity_map[y][x] = 1
                else:
                    for i in range(1, mismatch_range):
                        if x - i >= 0 and abs( self.left_disparity_map[y][x - i] - self.right_disparity_map[y][x - self.left_disparity_map[y][x]] ) <= 1 :
                            validity_map[y][x] = 2 # mismatch
                            break
                        if x + i < self.w and abs( self.left_disparity_map[y][x + i] - self.right_disparity_map[y][x - self.left_disparity_map[y][x]] ) <= 1 :
                            validity_map[y][x] = 2 # mismatch
                            break
                    if validity_map[y][x] == 0:
                        validity_map[y][x] = 3 # occlusion                    
        for y in range(self.h): 
            for x in range(self.w):
                if validity_map[y][x] == 3:
                    for i in range(1, x + 1):
                        if validity_map[y][x - i] == 1:
                            self.left_disparity_map[y][x] = self.left_disparity_map[y][x - i]
                            break
                elif validity_map[y][x] == 2:
                    median = 0
                    for east in range(1, self.w - x): 
                        if validity_map[y][x + east] == 1:
                            median += self.left_disparity_map[y][x + east]
                            break
                    for south in range(1, self.h - y): 
                        if validity_map[y + south][x] == 1:
                            median += self.left_disparity_map[y + south][x]
                            break
                    for west in range(1, x + 1): 
                        if validity_map[y][x - west] == 1:
                            median += self.left_disparity_map[y][x - west]
                            break
                    for north in range(1, y + 1): 
                        if validity_map[y - north][x] == 1:
                            median += self.left_disparity_map[y - north][x]
                            break
                    for northeast in range( min(self.w - 1 - x, y) ): 
                        if validity_map[y - northeast][x + northeast] == 1:
                            median += self.left_disparity_map[y - northeast][x + northeast]
                            break
                    for southeast in range( min(self.w - 1 - x, self.h - 1 - y) ): 
                        if validity_map[y + southeast][x + southeast] == 1:
                            median += self.left_disparity_map[y + southeast][x + southeast]
                            break
                    for southwest in range( min(x, self.h - 1 - y) ): 
                        if validity_map[y + southwest][x - southwest] == 1:
                            median += self.left_disparity_map[y + southwest][x - southwest]
                            break
                    for northwest in range( min(x, y) ): 
                        if validity_map[y - northwest][x - northwest] == 1:
                            median += self.left_disparity_map[y - northwest][x - northwest]
                            break
                    self.left_disparity_map[y][x] = int(median / 8)
        return self.left_disparity_map

    def SE(self, int_disparity_map, SGM_array_left):
#         h, w = int_disparity_map.shape
        se_disparity_map = np.zeros((self.h, self.w), dtype = 'uint8')
        for y in range(self.h):
            for x in range(self.w):
                d = int_disparity_map[y][x]
                c =  SGM_array_left[y][x][d]
                if d - 1 >= 0:
                    c_minus = SGM_array_left[y][x][d - 1]
                else:
                    c_minus = 0
                if d + 1 < self.dmax:
                    c_plus = SGM_array_left[y][x][d + 1]
                else:
                    c_plus = 0
                if 2 * (c_plus - 2 * c + c_minus) > 1e-5:
                    lilikoko = int( (c_plus - c_minus) / ( 2 * (c_plus - 2 * c + c_minus) ) )
                    if lilikoko >= -1 and lilikoko <= 1:
                        se_disparity_map[y][x] = int_disparity_map[y][x] - lilikoko
        return se_disparity_map

# def padding_3D(img, padding_size):  
#     pad = np.pad(img, ( (padding_size, padding_size), (padding_size, padding_size), (0, 0) ), 'constant')
#     return pad

# def main():
#     img_left_grayscale = cv2.imread('im3.png', 0).astype(int)
#     img_right_grayscale = cv2.imread('im4.png', 0).astype(int)
#     h,w = img_left_grayscale.shape
    
#     matching_cost_array = np.random.randn(h, w, dmax) 
#     matching_cost_array = np.pad(matching_cost_array, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
# #     matching_cost_array = padding_3D(matching_cost_array, cbca_distance - 1)
#     # output: (h + 13 * 2, w + 13 * 2, dmax)
    
#     model = Super_Solver(img_left_grayscale, img_right_grayscale, cbca_intensity, cbca_distance, cbca_num_iterations_1, dmax, sgm_p1, sgm_p2, sgm_q1, sgm_q2, sgm_d, sgm_v)
#     model.img_preprocessing()
#     model.compute_arm_length()
    
#     tic = time.time()
#     u_array_left, u_array_right = model.support_region_left(), model.support_region_right()
#     toc = time.time()
#     print('compute support region: ', toc - tic)

#     tic = time.time()                                                                                                                                 
#     matching_cost_array_left_before_SGM = model.compute_matching_cost_left(matching_cost_array, cbca_num_iterations_1)     
#     matching_cost_array_right_before_SGM = model.compute_matching_cost_right(matching_cost_array, cbca_num_iterations_1)
#     # output: (h + 2 * center, w + 2 * center, dmax, iteration + 1)  with padding
# #     print(matching_cost_array_left_before_SGM)
#     toc = time.time()
#     print('compute matching_cost_before_SGM: ', toc - tic)

#     tic = time.time()
#     SGM_array_left = model.SGM_left(matching_cost_array_left_before_SGM)
#     SGM_array_right = model.SGM_right(matching_cost_array_right_before_SGM)
#     # output: (h, w, dmax)
#     toc = time.time()
#     print('compute SGM: ', toc - tic)
    
#     SGM_padding_array_left = np.pad(SGM_array_left, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
#     SGM_padding_array_right = np.pad(SGM_array_right, ( (cbca_distance - 1, cbca_distance - 1), (cbca_distance - 1, cbca_distance - 1), (0, 0) ), 'constant')
# #     SGM_padding_array_left = padding_3D(SGM_array_left, cbca_distance - 1)
# #     SGM_padding_array_right = padding_3D(SGM_array_right, cbca_distance - 1)
#     tic = time.time()
#     matching_cost_array_left_after_SGM = model.compute_matching_cost_left(SGM_padding_array_left, cbca_num_iterations_2)
#     matching_cost_array_right_after_SGM = model.compute_matching_cost_right(SGM_padding_array_right, cbca_num_iterations_2)
#     # output: (h + 2 * center, w + 2 * center, dmax, iteration + 1)
#     toc = time.time()
#     print('compute matching_cost_after_SGM: ', toc - tic)

#     # de-pad
#     matching_cost_array_left_after_SGM = matching_cost_array_left_after_SGM[cbca_distance - 1 : h + cbca_distance - 1, cbca_distance - 1 : w + cbca_distance - 1, :, :]
#     matching_cost_array_right_after_SGM = matching_cost_array_right_after_SGM[cbca_distance - 1 : h + cbca_distance - 1, cbca_distance - 1 : w + cbca_distance - 1, :, :]
    

#     tic = time.time()
#     left_disparity_map = model.compute_left_disparity_map(matching_cost_array_left_after_SGM[:, :, :, cbca_num_iterations_2]).astype(int)
#     right_disparity_map = model.compute_right_disparity_map(matching_cost_array_right_after_SGM[:, :, :, cbca_num_iterations_2]).astype(int)
#     # output: (h, w)
#     toc = time.time()
#     print('compute disparity map: ', toc - tic)

#     tic = time.time()
#     int_disparity_map = model.check_validity()
#     # output: (h, w)
#     toc = time.time()
#     print('compute INT_disparity_map: ', toc - tic)

#     tic = time.time()
#     se_disparity_map = model.SE(int_disparity_map, SGM_array_left)
#     MB = cv2.medianBlur(se_disparity_map.astype('uint8'), 5)
#     blurred = cv2.bilateralFilter(MB, 5, 9, 16) 
#     # cv2.imwrite(...)
#     toc = time.time()
#     print('finish: ', toc - tic)
    
# if __name__ == '__main__':
#     main()


# In[13]:


# get_ipython().system('jupyter nbconvert --to script final.ipynb')


# In[ ]:




