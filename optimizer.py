
def parse_from_optimizer(parser):
	parser.add_argument("--GT")
	return parser

class optimizer():
	def __init__(self, args):
		self.args = args
		return
	def run(self, in_img):
		assert(in_img.shape[2] == self.args.maxDisp)
		# Run logic here
		#
		#
		assert(out_img.shape == in_img.shape[:2])
		return out_img