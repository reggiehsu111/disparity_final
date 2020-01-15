import numpy as np
from cv2.ximgproc import weightedMedianFilter
import cv2
from scipy import ndimage
from cv2.ximgproc import guidedFilter

def parse_from_refiner(parser):
    return parser

def form_color_map(disp):
    # normalize disparity to 0.0~1.0 for visualization
    max_disp = np.nanmax(disp[disp != np.inf])
    min_disp = np.nanmin(disp[disp != np.inf])
    disp_normalized = (disp - min_disp) / (max_disp - min_disp)

    # Jet color mapping
    disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
    disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    return disp_normalized

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
            out_img = self.refinement(D_l, D_r, *args)
        assert(len(out_img.shape)==2)
        return out_img

    def base_method(self, D_l, D_r, *args):
        """
            base method used in HW4
        """
        Il = args[0]
        CM_out_l = args[1]
        CM_out_r = args[2]
        gray = cv2.cvtColor(Il, cv2.COLOR_RGB2GRAY)
        h, w = D_l.shape
        D_l = self.border_refinement(D_l, h, w)
        D_r = self.border_refinement(D_r, h, w)
        D_l = cv2.medianBlur(D_l.astype('uint8'),3).astype('int')
        D_r = cv2.medianBlur(D_r.astype('uint8'),3).astype('int')
        D_l = self.edge_detection(D_l.astype(np.int32), CM_out_l, diff=5)
        D_r = self.edge_detection(D_r.astype(np.int32), CM_out_r, diff=5)
        # consistency check
        y, x = np.indices((h, w))
        outlier = self.find_outlier(D_l, D_r, h, w)
        # check_idx = outlier
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
        

        # labels = guidedFilter(guide=gray, src=labels.astype(np.uint8), radius=1, eps=50, dDepth=-1)


        labels = cv2.fastNlMeansDenoising(labels.astype(np.uint8))

        # labels = self.edge_detection(labels.astype(np.int32), CM_out_l, diff=5)

        # labels = self.subpixel_enhancement(labels.astype(np.int32), CM_out_l)

        outlier = self.find_outlier(D_l, D_r, h, w)
        labels = self.segmentation(Il, labels, outlier, 200, 200, True)

        labels = cv2.fastNlMeansDenoising(labels.astype(np.uint8))

        # labels = guidedFilter(guide=Il, src=labels.astype(np.uint8), radius=2, eps=30, dDepth=-1)
        disp_normalized = form_color_map(labels)
        cv2.imwrite('log/bilateralFilter.jpg', disp_normalized)
        return labels.astype(np.float32)

    def find_outlier(self, D_l, D_r, h, w, dilate=True):
        D_L = D_l.astype('int')
        D_R = D_r.astype('int')
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        outlier = np.zeros((h,w))
        for x in range(w):
            for y in range(h):
                if(x-D_L[y,x] >= 0 and x-D_L[y,x] < w):
                    if(np.abs(D_L[y,x]-D_R[y,x-D_L[y,x]]) < 1.5):
                        outlier[y,x] = 1
        return outlier
    def segmentation(self, Il, D_l, outlier, k, min_size, bi):
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=k, min_size=min_size)
        segment = segmentator.processImage(Il)   

        d = np.copy(D_l).astype('int')

        for i in range(np.max(segment)):
            valid = np.logical_and(outlier == 1, segment == i)
            invalid = np.logical_and(outlier == 0, segment == i)

            if(d[valid].size != 0 and d[invalid].size != 0 and d[valid].size/(d[valid].size+d[invalid].size) > 0.15):
                vote = np.bincount(d[valid])
                index = np.argwhere(vote == np.max(vote))
                mean = np.mean(d[valid])
                new_d = 255
                for j in index:
                    if j - mean < new_d:
                        new_d = j
                d[invalid] = new_d

            if(d[invalid].size/(d[valid].size+d[invalid].size) > 0.95 and np.max(np.where(np.logical_or(valid,invalid) == True)[1]) < np.max(D_l) + 5):
                d[segment == i] = int((self.max_disp) * 0.9)

        if bi:
            d = cv2.bilateralFilter(d.astype('uint8'),10,9,2).astype('float32')
        
        D_l = d
        return D_l

    def border_refinement(self, disp, h, w):
        img_valid = disp[2:h-2,5:w-2].astype('uint8')
        new_img = cv2.copyMakeBorder(img_valid,2,2,5,2,cv2.BORDER_REPLICATE)
        return new_img.astype('int')

    def subpixel_enhancement(self, labels, CM_out_l):
        h,w = labels.shape
        for y in range(h):
            for x in range(w):
                disp = labels[y, x]
                if disp >= 1 and disp < self.max_disp - 1:
                    denominator = 2 * (CM_out_l[disp - 1, y, x] + CM_out_l[disp + 1, y, x] - 2 * CM_out_l[disp, y, x])
                    if denominator > 1e-5:
                        labels[y, x] = disp - min(1, max(-1, (CM_out_l[disp + 1, y, x] - CM_out_l[disp - 1, y, x]) / denominator))
        return labels

    def edge_detection(self, labels, cost, diff=5):
        h, w = labels.shape

        result = np.zeros_like(labels)
        dx = ndimage.filters.sobel(labels, axis=0)
        dy = ndimage.filters.sobel(labels, axis=1)

        for y in range(h):
            for x in range(w):
                result[y, x] = labels[y, x]
                if dx[y, x] > diff and x > 0 and x < w - 1:
                    if cost[labels[y, x - 1], y, x] < cost[labels[y, x], y, x] and cost[labels[y, x + 1], y, x] < cost[labels[y, x], y, x]:
                        if abs(cost[labels[y, x - 1], y, x] - cost[labels[y, x], y, x]) < abs(cost[labels[y, x + 1], y, x] - cost[labels[y, x], y, x]):
                            result[y, x] = labels[y, x - 1]
                        else:
                            result[y, x] = labels[y, x + 1]
                    elif cost[labels[y, x + 1], y, x] < cost[labels[y, x], y, x]:
                        result[y, x] = labels[y, x + 1]
                    else:
                        result[y, x] = labels[y, x - 1]
                
                if dy[y, x] > diff and y > 0 and y < h - 1:
                    if cost[labels[y - 1, x], y, x] < cost[labels[y, x], y, x] and cost[labels[y + 1, x], y, x] < cost[labels[y, x], y, x]:
                        if abs(cost[labels[y - 1, x], y, x] - cost[labels[y, x], y, x]) < abs(cost[labels[y + 1, x], y, x] - cost[labels[y, x], y, x]):
                            result[y, x] = labels[y - 1, x]
                        else:
                            result[y, x] = labels[y + 1, x]
                    elif cost[labels[y + 1, x], y, x] < cost[labels[y, x], y, x]:
                        result[y, x] = labels[y + 1, x]
                    else:
                        result[y, x] = labels[y - 1, x]
        
        return result


    def refinement(self, displ, dispr, *args):
        threshold = 0.005   # determine the holes need to be filled with
        b_radius = 9   # radius for bilateralFilter
        w_radius = 32   # radius for weightedMedianFilter

        Il = args[0]
        CM_out_l = args[1]

        # left-right consistency check
        occluded = np.zeros_like(displ) # 0 => unoccluded but unstable, 1 => occluded, 2 => unoccluded and stable
        for i in range(len(displ)):
            for j in range(len(displ[i])):
                # separate occluded and unoccluded
                if abs(displ[i, j] - dispr[i, j - displ[i, j]]) >= 1:
                    occluded[i, j] = 1

        # threshold hole filling
        stable = np.zeros_like(displ)
        occlusion = np.zeros_like(displ)
        unstable = np.zeros_like(displ)
        for i in range(len(displ)):
            for j in range(len(displ[i])):
                disp, cost = [], []
                # unoccluded, include stable and unstable
                if occluded[i,j] == 0:
                    c1, c2 = np.sort(CM_out_l[:, i, j])[:2]
                    # stable
                    if (c2 - c1) >= threshold * c2 and c1 != c2:
                        occluded[i,j] = 2
                        stable[i,j] = displ[i,j]
                    # unstable => fill the hole
                    else:
                        unstable[i,j] = displ[i,j]

                # occluded
                else:
                    occlusion[i,j] = displ[i,j]

        # labels = ndimage.median_filter(labels, 15)        
        # labels = cv2.bilateralFilter(labels.astype(np.uint8), b_radius, 1, 3)
        # labels = cv2.ximgproc.jointBilateralFilter(joint=Il.astype(np.uint8), src=labels.astype(np.uint8), d=11, sigmaColor=1, sigmaSpace=1)
        # labels = BF.joint_bilateral_filter(labels, Il.astype(np.uint8), occluded)

        # cv2.imshow('disp', (displ.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imshow('dispr', (dispr.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)

        # cv2.imshow('stable', (stable.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imshow('unstable', (unstable.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imshow('occluded', (occlusion.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)

        BF = Joint_bilateral_filter(3, 1, 3, border_type='reflect')
        _Il, _occluded = Il.copy(), occluded.copy()
        for disp in range(len(CM_out_l)):
            print('compute bilateral filter:', disp)
            # shift Il and occluded right by disp
            if disp != 0:
                _Il[:, :disp, :], _occluded[:, :disp] = 2, 2

            # cv2.imshow('cost volumn', (CM_out_l[disp].astype(np.float32)/np.max(CM_out_l)*255).astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.imshow('Il', _Il.astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.imshow('occluded', (_occluded.astype(np.float32)/2*255).astype(np.uint8))
            # cv2.waitKey(0)

            # apply joint bilateral filter on cost volumn at the occlusion points and unstable points only
            CM_out_l[disp] = BF.joint_bilateral_filter(CM_out_l[disp], _Il, _occluded, displ)
            
            # cv2.imshow('cost volumn out', (CM_out_l[disp].astype(np.float32)/np.max(CM_out_l)*255).astype(np.uint8))
            # cv2.waitKey(0)

        # apply Winner Take All on cost volumn again 
        labels = CM_out_l.argmin(axis=0)

        cv2.imshow('WTA', (labels.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        cv2.waitKey(0)

        # weighted median filter
        labels_filtered = cv2.ximgproc.weightedMedianFilter(joint=Il.astype(np.uint8), src=labels.astype(np.uint8), r=10, sigma=15)
        labels = np.where(occluded == 2, labels, labels_filtered)

        # cv2.imshow('Weighted median filter', (labels.astype(np.float32)/self.max_disp*255).astype(np.uint8))
        # cv2.waitKey(0)

        return labels.astype(np.float32)

    def print_v(self, message):
        if self.args.verbose:
            print(message)
        return

class Joint_bilateral_filter(object):
    def __init__(self, radius, sigma_s, sigma_r, border_type='reflect'):
        self.border_type = border_type
        self.radius = radius
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def joint_bilateral_filter(self, input, guidance, occluded, displ):
        h, w = input.shape
        output = input.copy()
        output.astype("float64")

        # padding
        r = int(self.radius)
        displ = cv2.copyMakeBorder(displ, r, r, r, r, cv2.BORDER_REFLECT)
        input = cv2.copyMakeBorder(input, r, r, r, r, cv2.BORDER_REFLECT)
        guidance = cv2.copyMakeBorder(guidance, r, r, r, r, cv2.BORDER_REFLECT)
        input = input.astype("float64")
        guidance = guidance.astype("int64")

        # pre-compute kernel
        range_kernel_table = np.exp( - (np.arange(256) / 255)**2 / (2 * self.sigma_r * self.sigma_r) )
        
        x, y = np.meshgrid( np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r )
        spatial_kernel_table = np.exp( -(x * x + y * y) / (2 * self.sigma_s * self.sigma_s) )

        # bilateral filter compute
        if input.ndim == 2:
            for i in range(r, r + h):
                for j in range(r, r + w):
                    if occluded[i - r, j - r] != 2:   # unstable or occluded
                        kernel = spatial_kernel_table * \
                                (range_kernel_table[np.abs(guidance[i - r:i + r + 1, j - r:j + r + 1, 0] - guidance[i, j, 0])] *
                                range_kernel_table[np.abs(guidance[i - r:i + r + 1, j - r:j + r + 1, 1] - guidance[i, j, 1])] *
                                range_kernel_table[np.abs(guidance[i - r:i + r + 1, j - r:j + r + 1, 2] - guidance[i, j, 2])])
                        kernel /= np.sum(kernel) 

                        if occluded[i - r, j - r] == 1:    # occluded
                            if len(displ[i - r:i + r + 1, j - r:j + r + 1][displ[i - r:i + r + 1, j - r:j + r + 1] != 0]):
                                Dmin = np.min(displ[i - r:i + r + 1, j - r:j + r + 1][displ[i - r:i + r + 1, j - r:j + r + 1] != 0])
                                disparity_kernel = np.exp(- np.abs(displ[i-r, j-r] - Dmin) / (0.5 * Dmin) )
                                kernel *= disparity_kernel
                        
                        output[i - r, j - r] = np.sum(kernel * input[i - r:i + r + 1, j - r:j + r + 1])

        return output.astype(np.uint8)