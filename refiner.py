
def parse_from_refiner(parser):
	return parser


class refiner():
	def __init__(self, args):
		self.args = args
		return
	def run(self, in_img):
		assert(len(in_img.shape)==2)
		# Run logic here
		assert(len(out_img.shape)==2)
		return out_img