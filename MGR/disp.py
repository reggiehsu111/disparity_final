from .optimizer import *
from .refiner import *
from .costmgr import *
import time
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def parse_from_disp(parser):
    parser.add_argument('--CM_base', action='store_true', help='Specify only when using base method for costMgr')
    parser.add_argument('--OP_base', action='store_true', help='Specify only when using base method for optimizer')
    parser.add_argument('--RF_base', action='store_true', help='Specify only when using base method for refiner')
    return parser

class dispMgr():
    def __init__(self, args):
        self.CM = costMgr(args)
        self.OP = optimizer(args)
        self.RF = refiner(args)
        self.args = args

    def computeDisp(self, Il, Ir):
        Il = Il.astype(np.float32)
        Ir = Ir.astype(np.float32)

        # Cost computation 
        self.print_v("##### Computing Cost... #####")
        start = time.time()
        CM_out_l, CM_out_r = self.CM.run(Il, Ir, base=self.args.CM_base)
        self.print_v("##### Elapsed time: "+ str(time.time()-start) +" #####\n")

        # Optimization
        self.print_v("##### Optimizing... #####")
        start = time.time()
        OP_out_l, OP_out_r = self.OP.run(CM_out_l, CM_out_r, base=self.args.OP_base)
        self.print_v("##### Elapsed time: "+ str(time.time()-start) +" #####\n")

        print("Writing out...")
        cv2.imwrite("log/OP_l.jpg", OP_out_l*3)
        cv2.imwrite("log/OP_r.jpg", OP_out_r*3)

        # Refinement
        self.print_v("##### Refining... #####")
        start = time.time()
        disp = self.RF.run(OP_out_l, OP_out_r, Il, CM_out_l, CM_out_r, base=self.args.RF_base)
        self.print_v("##### Elapsed time: "+ str(time.time()-start) +" #####\n")

        return disp

    def print_v(self, message):
        if self.args.verbose:
            print(message)
        return