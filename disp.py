from optimizer import *
from refiner import *
from costmgr import *

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

        CM_out_l, CM_out_r = self.CM.run(Il, Ir, base=self.args.CM_base)
        OP_out_l = self.OP.run(CM_out_l, base=self.args.OP_base)
        OP_out_r = self.OP.run(CM_out_r, base=self.args.OP_base)
        disp = self.RF.run(OP_out_l, OP_out_r, Il, base=self.args.RF_base)
        return disp