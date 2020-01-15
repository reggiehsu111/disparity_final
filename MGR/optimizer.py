import numpy as np

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
        out_img_l_true = out_img_l.copy()
        out_img_r_true = out_img_r.copy()
        wl, hl = out_img_l.shape
        wr, hr = out_img_r.shape
        windows = 2
        check1 = 0
        check2 = 0
        for i in range(windows,wl-windows):
            for j in range(windows,hl-windows):
                #first check if it is bad
                deltaup    = abs(out_img_l[i][j]-out_img_l[i][j+1])
                deltadown  = abs(out_img_l[i][j]-out_img_l[i][j-1])
                deltaleft  = abs(out_img_l[i][j]-out_img_l[i-1][j])
                deltaright = abs(out_img_l[i][j]-out_img_l[i+1][j])
                maxdis = 10
                if(deltaup>maxdis and deltadown>maxdis and deltaleft>maxdis and deltaright>maxdis):
                    vote = np.zeros(self.args.max_disp + 1)
                    for k in range(-2,3):
                        for l in range(-2,3):
                            vote[out_img_l[i+k][j+l]] += 1
                    out_img_l_true[i][j] = vote.argmax()
                    check1 += 1
        for i in range(windows,wr-windows):
            for j in range(windows,hr-windows):
                #first check if it is bad
                deltaup    = abs(out_img_r[i][j]-out_img_r[i][j+1])
                deltadown  = abs(out_img_r[i][j]-out_img_r[i][j-1])
                deltaleft  = abs(out_img_r[i][j]-out_img_r[i-1][j])
                deltaright = abs(out_img_r[i][j]-out_img_r[i+1][j])
                maxdis = 10
                if(deltaup>maxdis and deltadown>maxdis and deltaleft>maxdis and deltaright>maxdis):
                    vote = np.zeros(self.args.max_disp + 1)
                    for k in range(-2,3):
                        for l in range(-2,3):
                            vote[out_img_r[i+k][j+l]] += 1
                    out_img_r_true[i][j] = vote.argmax()
                    check2 += 1
        print(check1)      
        print(check2)
        return out_img_l_true, out_img_r_true

    def improved_method(self, in_img_l, in_img_r):
        out_img_l = np.zeros((h, w))
        out_img_r = np.zeros((h, w))
        return out_img_l, out_img_r
