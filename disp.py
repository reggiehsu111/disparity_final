from optimizer import *
from refiner import *
from costmgr import *


class dispMgr():
    def __init__(self, args):
        self.CM = costMgr(args)
        self.OP = optimizer(args)
        self.RF = refiner(args)

    def computeDisp(self, Il, Ir):
        Il = Il.astype(np.float32)
        Ir = Ir.astype(np.float32)

        CM_out_l, CM_out_r = self.CM.run(Il, Ir, base=True)
        OP_out_l = self.OP.run(CM_out_l, base=True)
        OP_out_r = self.OP.run(CM_out_r, base=True)
        disp = self.RF.run(OP_out_l, OP_out_r, Il, base=True)
        return disp