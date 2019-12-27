import numpy as np

def parse_from_optimizer(parser):
    return parser

class optimizer():
    def __init__(self, args):
        self.args = args
        return

    def run(self, in_img, *args, base=False):
        """
            Params:
                Input shape  : (max_disp+1, h, w)
                *args        : Anything to pass in
                base         : if bas method is used 
            Output shape : (h, w)
            
        """
        assert(in_img.shape[0] == self.args.max_disp+1)
        if base:
            out_img = self.base_method(in_img)
        else:
            # Write you method here
            _, h, w = in_img.shape
            # Change this line of code
            out_img = np.zeros((h, w))
        assert(out_img.shape == in_img.shape[1:])
        return out_img

    def base_method(self, in_img):
        """
            base method used in HW4
        """
        out_img = in_img.argmin(axis=0)
        return out_img