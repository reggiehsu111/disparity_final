import numpy as np
import cv2
import math


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):

        if border_type == 'reflect':
            self.border_type = 'symmetric'
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.r = 3*sigma_s

        self.exp_table_s = {}
        self.exp_table_r = {}
        self.map_s = np.arange(0, self.r+1, dtype=np.float64)
        self.map_r = np.arange(0, 256, dtype=np.float64)
        self.init_exp_table()

        self.spatial_kernel = self.get_spatial_kernel()

    def init_exp_table(self):
        for i in range(256):
            if i <= self.r:
                self.exp_table_s[i] = math.exp(-i**2/(2 * self.sigma_s**2))
            self.exp_table_r[i] = math.exp(-(i/255)**2/(2 * self.sigma_r**2))
        self.map_s[list(self.exp_table_s.keys())] = list(self.exp_table_s.values())
        self.map_r[list(self.exp_table_r.keys())] = list(self.exp_table_r.values())

    def get_gaussian(self, arr, table='r'):
        exp_map = self.map_r if table == 'r' else self.map_s
        arr = np.abs(arr)
        arr = exp_map[arr]
        return arr

    def get_spatial_kernel(self):
        ys, xs = np.meshgrid(np.arange(-self.r, self.r+1), np.arange(-self.r, self.r+1))
        # return np.exp(-(ys**2 + xs**2) / (2 * self.sigma_s**2))
        return self.get_gaussian(ys, table='s') * self.get_gaussian(xs, table='s')

    def get_range_kernel(self, center, img):
        center_y, center_x = center
        if img.ndim == 3:
            intensity_diff = img[center_y - self.r:center_y + self.r+1, center_x - self.r:center_x + self.r+1, :] - \
                             img[center_y, center_x, :]
            # kernel = np.exp(-(intensity_diff**2).sum(axis=2) / (2 * self.sigma_r**2))
            kernel_rows = self.get_gaussian(intensity_diff.astype(int))
            kernel = kernel_rows[:, :, 0] * kernel_rows[:, :, 1] * kernel_rows[:, :, 2]
        else:
            intensity_diff = img[center_y - self.r:center_y + self.r+1, center_x - self.r:center_x + self.r+1] - \
                             img[center_y, center_x]
            # kernel = np.exp(-(intensity_diff**2) / (2 * self.sigma_r**2))
            kernel = self.get_gaussian(intensity_diff.astype(int))
        return kernel

    def joint_bilateral_filter(self, src, guidance):
        # TODO
        h, w, c = src.shape
        src_padded = np.pad(src, ((self.r,), (self.r,), (0,)), mode=self.border_type).astype(np.float64)
        src_filtered = np.zeros_like(src, dtype=np.float64)
        # guidance = guidance/255
        if guidance.ndim == 3:
            guidance_padded = np.pad(guidance, ((self.r,), (self.r,), (0,)), mode=self.border_type)
        else:
            guidance_padded = np.pad(guidance, ((self.r,), (self.r,)), mode=self.border_type)
        for y in range(self.r, h+self.r):
            for x in range(self.r, w+self.r):
                range_kernel = self.get_range_kernel((y, x), guidance_padded)
                kernel = self.spatial_kernel * range_kernel
                kernel = kernel / kernel.sum()
                kernel = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)
                src_filtered[y-self.r, x-self.r] = (kernel * src_padded[y-self.r:y+self.r+1, x-self.r:x+self.r+1])\
                    .sum(axis=0).sum(axis=0)
        return src_filtered

    def get_jbl_weights(self, guidance, d_mask, valid_mask):
        # TODO
        h, w, _ = guidance.shape
        mask = np.logical_and(d_mask, valid_mask)
        weights = np.zeros((h, w))

        mask_padded = np.pad(mask, ((self.r,), (self.r,)), mode=self.border_type)
        # guidance = guidance/255
        if guidance.ndim == 3:
            guidance_padded = np.pad(guidance, ((self.r,), (self.r,), (0,)), mode=self.border_type)
        else:
            guidance_padded = np.pad(guidance, ((self.r,), (self.r,)), mode=self.border_type)
        for y in range(self.r, h + self.r):
            for x in range(self.r, w + self.r):
                mask_part = mask_padded[y-self.r:y+self.r+1, x-self.r:x+self.r+1]
                if valid_mask[y-self.r, x-self.r] == 1 or not mask_part.any():
                    continue

                range_kernel = self.get_range_kernel((y, x), guidance_padded)
                kernel = self.spatial_kernel * range_kernel * mask_part

                weights[y - self.r, x - self.r] = (kernel.sum() / np.count_nonzero(mask_part))
        return weights
