import cv2
import numpy as np
from scipy import sqrt, pi, arctan2
from scipy.ndimage import uniform_filter


def HoG(img, cell_per_blk=(3,3), pix_per_cell=(8, 8), orientation = 3):
	if img is None:
		print " pic read failed"
		return -1

	if img.ndim > 3:
		print " gray-scale process only for speed performance"
		return -1

	# gradient computation
	gradient_x =np.zeros(img.shape)
	gradient_y =np.zeros(img.shape)
	gradient_x[:, :-1] = np.diff(img, n=1, axis=1)
	gradient_y[:-1, :] = np.diff(img, n=1, axis=0)
	magnitude = sqrt(gradient_x ** 2 + gradient_y ** 2)
	ori = arctan2(gradient_y, (gradient_x + 1e-15)) * (180 / pi) + 90
	
	# Orientation Binning
	img_h,img_w = img.shape
	cx, cy = pix_per_cell
	bx, by = cell_per_blk
	ncell_x = int(np.floor(img_w//cx))
	ncell_y = int(np.floor(img_h//cy))
	ori_histogram = np.zeros((ncell_y,ncell_x,orientation))
	for i in range(0, orientation):
		temp1 = np.where(ori < 180 / orientation * (i + 1), ori, 0)
		temp1 = np.where(ori >= 180 / orientation * i,temp1, 0)
		temp2 = np.where(temp1>0, magnitude, 0)
		ori_histogram[:,:,i] = uniform_filter(temp2, size=(cy,cx))[cy/2::cy, cx/2::cx]

	# normalization
	n_blocksx = (ncell_x - bx) + 1
	n_blocksy = (ncell_y - by) + 1
	normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientation))
	for x in range(n_blocksx):
		for y in range(n_blocksy):
			block = ori_histogram[y:y + by, x:x + bx, :]
			eps = 1e-5
			normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)

	return normalised_blocks


if __name__ == '__main__':
	pic = './pics/img1.jpg'
	img = cv2.imread(pic,0)
	hog = HoG(img)
	print hog




