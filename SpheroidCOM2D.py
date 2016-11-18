# -*- coding: utf-8 -*-
"""
SheroidCOM2D: Center of mass of single plane spheroid images
Created on Mon Oct 03 10:23:14 2016

@author: Fanny Georgi, based on https://github.com/cfinch/Shocksolution_Examples/tree/master/PairCorrelation

comment legend:
    ################## funtion of code below
    ### error
"""


################## clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from scipy import ndimage
import math
import cv2
from scipy import ndimage
import skimage
import skimage.io
from skimage import measure,morphology

################## define images
nucleiImage = 'B01_405_z720.tif'
virusImage = 'B01_488_z720.tif'

nucleiIm = cv2.imread(nucleiImage, -1) #0 = grey, 1=RGB, -1=unchanged
virusIm = cv2.imread(virusImage, -1) #0 = grey, 1=RGB, -1=unchanged
## same as 
#nucleiIm = skimage.io.imread(nuceiImage, plugin='tifffile')
## or
#nucleiImArray=numpy.array(nucleiIm)

plt.subplot(131)
plt.imshow(nucleiIm)
plt.set_cmap('gray')
plt.axis('on') 
plt.subplot(132)
plt.imshow(virusIm)
plt.set_cmap('gray')
plt.axis('on')
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
plt.show()

################## convert to 8bit
#nucleiIm8bit = (nucleiIm*255).astype('uint8')
#nucleiIm8bit = numpy.asarray(nucleiIm8bit)
#plt.imshow(nucleiIm8bit)
#plt.set_cmap('gray')
#plt.axis('on')

################## gaussian smoothing
n = 8
l = 256

gaussian = ndimage.gaussian_filter(nucleiIm, sigma=l/(4.*n))

mask = (gaussian > gaussian.mean()).astype(numpy.float)
gaussianMean = gaussian.mean()
#mask = (gaussian > 0.25).astype(numpy.float)
#mask += 0.1 * gaussian
gaussian_img = mask + 0.2*numpy.random.randn(*mask.shape)

hist, bin_edges = numpy.histogram(gaussian_img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

binary_gaussian = gaussian_img > 0.5

################## dilation
# Remove small white regions
open_img = ndimage.binary_opening(binary_gaussian)
# Remove small black hole
close_img = ndimage.binary_closing(open_img)

dilation = numpy.abs(mask - close_img).mean() 

label_dilation, nb_labels = ndimage.label(mask)
print nb_labels # how many regions?

sizes = ndimage.sum(mask, label_dilation, range(nb_labels + 1))
mean_vals = ndimage.sum(close_img, label_dilation, range(1, nb_labels + 1))

plt.subplot(131)
plt.imshow(nucleiIm)
plt.set_cmap('gray')
plt.axis('on') 
plt.subplot(132)
plt.imshow(mask)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(binary_gaussian)
plt.set_cmap('gray')
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
plt.show()

#plt.subplot(131)
#plt.imshow(open_img)
#plt.set_cmap('gray')
#plt.axis('on') 
#plt.subplot(132)
#plt.imshow(close_img)
#plt.set_cmap('gray')
#plt.axis('off')
#plt.subplot(133)
#plt.imshow(label_dilation)
#plt.set_cmap('gray')
#plt.axis('off')
#plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
#plt.show()

#plt.imsave('mask.tif', mask, format='tif')

##################This does not work nicely
## Two subplots, unpack the axes array immediately
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.plot(x, y)
#ax1.set_title('Sharing Y axis')
#ax2.scatter(x, y)
#
#f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True)
#ax1.imshow(mask)
#ax2.imshow(gaussian_img)
#ax3.imshow(binary_gaussian)
#ax4.imshow(dilation_img)
#ax5.imshow(label_dilation) 
#plt.show().set_aspect('equal', adjustable='datalim')
#plt.show()

################# erosion
#eroded_img = ndimage.binary_erosion(binary_gaussian)
#reconstruct_img = ndimage.binary_propagation(eroded_img, mask=binary_gaussian)
#tmp = numpy.logical_not(reconstruct_img)
#eroded_tmp = ndimage.binary_erosion(tmp)
#reconstruct_final = numpy.logical_not(ndimage.binary_propagation(eroded_tmp, mask=tmp))
#
#erosion = numpy.abs(mask - reconstruct_final).mean() 
#
#plt.subplot(131)
#plt.imshow(reconstruct_final)
#plt.set_cmap('gray')
#plt.axis('on')
#plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
#plt.show() 

################# Morphosphere's dilation
dilationDisk = 10
blockSize = 111

processedImage = (nucleiIm*255).astype('uint8')

thresholdedImage = cv2.adaptiveThreshold(processedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,0) #return inverted threshold, 0 = threshold correction factor
selem = skimage.morphology.disk(dilationDisk)

thresholdedImage = cv2.dilate(thresholdedImage,selem,iterations = 1)

thresholdedImage = ndimage.binary_fill_holes(thresholdedImage)

labeledImage = measure.label(thresholdedImage)
allProperties = measure.regionprops(labeledImage)

imageHeight, imageWidth = thresholdedImage.shape[:2]
#initialize empty arrays for area and distance filtering
areas=numpy.empty(len(allProperties))
distancesToCenter = numpy.empty(len(allProperties))
labels = numpy.empty(len(allProperties))

plt.subplot(131)
plt.imshow(processedImage)
plt.set_cmap('gray')
plt.axis('on') 
plt.subplot(132)
plt.imshow(mask)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(thresholdedImage)
plt.set_cmap('gray')
plt.axis('off')
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)
plt.show()

################## calculate center of mass
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.measurements.center_of_mass.html
centerOfMass_nuceiIm = ndimage.measurements.center_of_mass(nucleiIm)
#centerOfMass_mask = ndimage.measurements.center_of_mass(mask)
#centerOfMass_dilation_img = ndimage.measurements.center_of_mass(dilation_img)

################## convert to 8 bit for displaying
## none of this worked 
#nucleiIm8bit = skimage.util.img_as_ubyte(nucleiImage) #convert to unsigned 8bit
#nucleiIm8bit = (round(nucleiIm*255))
#nuceliIm8bit = cv2.cvtColor(nucleiIm, cv2.CV_8UC1) 

#cv2.imwrite('8bit.tif', nucleiIm8bit)
## alternatively
#plt.imsave('8bit.tif', nucleiIm8bit, format='tif')