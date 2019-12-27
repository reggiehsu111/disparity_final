import sys
from util import readPFM, writePFM, cal_avgerr
import numpy as np
import cv2

def main():
	# read disparity pfm file (float32)
	# the ground truth disparity maps may contain inf pixels as invalid pixels
	disp = readPFM(str(sys.argv[1]))

	# normalize disparity to 0.0~1.0 for visualization
	max_disp = np.nanmax(disp[disp != np.inf])
	min_disp = np.nanmin(disp[disp != np.inf])
	disp_normalized = (disp - min_disp) / (max_disp - min_disp)

	# Jet color mapping
	disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
	disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
	cv2.imshow("visualized disparity", disp_normalized)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def compare(images, output_str):
	normalized_images = []
	count = 0
	for disp in images:
		max_disp = np.nanmax(disp[disp != np.inf])
		min_disp = np.nanmin(disp[disp != np.inf])
		disp_normalized = (disp - min_disp) / (max_disp - min_disp)
		disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
		disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
		normalized_images.append(disp_normalized)
		count+=1
	horizontal_stacked = np.hstack(normalized_images)
	cv2.imshow(output_str, horizontal_stacked)
	print("Press Q to leave")
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return
# Compare ground truth with produced
def compare_GT(GT_path):
	disp = readPFM(str(sys.argv[1]))
	GT = readPFM(GT_path)
	error = np.abs(disp-GT)
	output_str = "disp, GT, errorq"
	compare([disp, GT, error], output_str)
	return

if __name__ == '__main__':
	# main()
	GT_path = sys.argv[2]
	compare_GT(GT_path)
	exit()