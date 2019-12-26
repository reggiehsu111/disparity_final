import numpy as np
import cv2
from disparity import disp_calculator as dc
from disparity import subpixel_enhancement as sub
from jerry import *
from optimizer import *
from refiner import *
from costmgr import *


class dispMgr():
	def __init__(self, args):
		self.maxDisp = args.maxDisp
		self.CM = costMgr(args)
		self.OP = optimizer(args)
		self.RF = refiner(args)

	def computeDisp(self, Il, Ir):
		CM_out_l, CM_out_r = self.CM.run(Il, Ir)
		OP_out_l = self.OP.run(CM_out_l)
		OP_out_r = self.OP.run(CM_out_r)
		disp = self.RF.run(OP_out_l, OP_out_r)
		return disp