import numpy as np
import cv2

def parse_from_optimizer(parser):
    return parser

class optimizer():
    def __init__(self, args):
        self.args = args
        return

    def run(self, in_img_l, in_img_r, *args, base=False):
        """
            Params:
                Input shape  : (max_disp+1, h, w)
                *args        : Anything to pass in
                base         : if base method is used 
            Output shape : (h, w)
            
        """
        assert(in_img_l.shape[0] == self.args.max_disp+1)
        assert(in_img_r.shape[0] == self.args.max_disp+1)
        if base:
            out_img_l, out_img_r = self.base_method(in_img_l, in_img_r)
        else:
            # Write you method here
            _, h, w = in_img_l.shape
            # Change this line of code
            out_img_l, out_img_r = self.improved_method(in_img_l, in_img_r)
        assert(out_img_l.shape == in_img_l.shape[1:])
        assert(out_img_r.shape == in_img_r.shape[1:])
        return out_img_l, out_img_r 

    def base_method(self, in_img_l, in_img_r):
        """
            base method used in HW4
        """
        out_img_l = in_img_l.argmin(axis=0)
        out_img_r = in_img_r.argmin(axis=0)
        return out_img_l, out_img_r

    def softmax(self, in_img):
        exp = np.exp(in_img)
        out_img = exp/np.sum(exp, axis = 0)
        return out_img

    def improved_method(self, in_img_l, in_img_r):
        temp_l = np.argsort(in_img_l, axis=0)
        temp_r = np.argsort(in_img_r, axis=0)
        # for x in range(temp_l.shape[0]):
        #     cv2.imwrite("log/disp_sorted/"+str(x)+"_l.jpg", temp_l[x])
        #     cv2.imwrite("log/disp_sorted/"+str(x)+"_r.jpg", temp_r[x])

        out_img_l = in_img_l.argmin(axis=0)
        out_img_r = in_img_r.argmin(axis=0)
        print("Denoising disparity map...")
        out_img_l = cv2.fastNlMeansDenoising(out_img_l.astype(np.uint8))
        out_img_r = cv2.fastNlMeansDenoising(out_img_r.astype(np.uint8))
        return out_img_l, out_img_r
