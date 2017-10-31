#!/usr/bin/env python

import math
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import skimage
from scipy import ndimage as ndi
from pylab import array, arange, uint8
from scipy.interpolate import griddata

def regionSeg(im):
	kernel = np.ones((8,8), np.uint8)
	kernel2 = np.ones((5,5), np.uint8)
	blurim = cv2.GaussianBlur(im,(5,5),3)
	enim = enhance(blurim)
	elevation_map = skimage.filters.sobel(enim)
	markers = np.zeros_like(enim)
	markers[enim < 30] = 1
	markers[enim > 150] = 2
	seg = skimage.morphology.watershed(elevation_map, markers)
	label_objects, nb_labels = ndi.label(seg-1)
	sizes = np.bincount(np.ravel(label_objects))
	mask_sizes = sizes > 200
	mask_sizes[0] = 0
	msk = mask_sizes[label_objects]
#msk = ndi.binary_fill_holes(msk)
	msk = np.array(msk, dtype = np.uint8)
#msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
#msk = msk.astype(int)
	return msk

def enhance(im):
	phi = 1
	theta = 4
	maxIntensity = 255
	x = arange(maxIntensity)
	newImage = (maxIntensity/phi)*(im/(maxIntensity/theta))**0.05
	newImage = array(newImage, dtype = uint8)
	return newImage

def polarinterp(theta):
	ax = np.linspace(-200,200,401)
	x = ax*math.cos(theta*math.pi/180)
	y = ax*math.sin(theta*math.pi/180)
	return x,y

def main():
	path = '/shared/s2/users/zdai/oct-models/recon/images/'
	imread_color = cv2.IMREAD_GRAYSCALE
	index = 0
	I = np.zeros((85,401,21))
	for imname in os.listdir(path):
		impath = os.path.join(path, imname)
		im = cv2.imread(impath, imread_color)
		im = cv2.resize(im, (401,267))
		mask = regionSeg(im)
		x,y = np.where(mask == 1)
		up = max(x)+5
		down = up - 85
		for i in range(max(y)):
			line = mask[:,i]
			x = np.where(line==1)
			a = min(x[0])
			b = max(x[0])
			mask[a:b,i] = np.ones(1,b-a+1)
		im = im*mask
		im = im[down:up,:]
		I[:,:,index] = im
		index += 1
		"""
		plt.imshow(im, cmap = 'Greys_r')
		plt.show()
		"""
	print(I.shape)
	theta = 180
	dtheta = 180/21
	m = range(-200,200,1)
	n = range(-200,200,1)
	X,Y = np.meshgrid(m,n)
	x,y,sli= [],[],[]
	for i in xrange(85):
		s = I[i,:,:]
		x,y,sli= [],[],[]
		for j in xrange(21):
			m,n = polarinterp(dtheta*j)
			x = np.append(x,m)
			y = np.append(y,n)
			sli = np.append(sli,s[:,j])
		Ti = griddata((x,y),sli,(X,Y),method = 'cubic')
		plt.imshow(Ti, cmap = 'Greys_r')
		plt.show()

	
if __name__ == '__main__':
	main()
