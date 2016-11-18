# -*- coding: utf-8 -*-
"""
Thresholding single plane input images 

Created on Mon Oct 03 15:36:40 2016
@author: Fanny Georgi, based on Vardan Andriasyan's MorphoSphere segmentation

comment legend:
    ################## funtion of code below
    ### error
"""

import skimage
import math
import cv2
from scipy import ndimage
from skimage import measure,morphology
import numpy as np
import matplotlib.pyplot as plt


################## open single z plane from 16bit image
nucleiImage = 'B01_405_z720.tif'
inputImage = cv2.imread(nucleiImage, -1) #0 = grey, 1=RGB, -1=unchanged

################## validate input visually
plt.imshow(inputImage)
plt.set_cmap('gray')
plt.axis('on')
plt.show()

################## convert to 8bit
processedImage = (inputImage*255).astype('uint8')
processedImage = np.asarray(processedImage)

minSpheroidArea = 500
dilationDisk = 100
blockSize = 501

thresholdedImage = cv2.adaptiveThreshold(processedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize,0) #0 = threshold correction factor
selem = skimage.morphology.disk(dilationDisk)

thresholdedImage = cv2.dilate(thresholdedImage,selem,iterations = 1)

thresholdedImage = ndimage.binary_fill_holes(thresholdedImage)

labeledImage = measure.label(thresholdedImage)
allProperties = measure.regionprops(labeledImage)


imageHeight, imageWidth = thresholdedImage.shape[:2]
################## initialize empty arrays for area and distance filtering
areas=np.empty(len(allProperties))
distancesToCenter = np.empty(len(allProperties))
labels = np.empty(len(allProperties))

################## find the index connected area which is closest to center of the image also area filter
i = 0
for props in allProperties:
    y0, x0 = props.centroid
    distance = math.sqrt((y0 -imageHeight/2)**2 + (x0 -imageWidth/2)**2)
    distancesToCenter[i] = distance
    areas[i] = props.area
    labels[i] = props.label
    i=i+1

################## filter by area
distancesToCenter = distancesToCenter[areas>minSpheroidArea]
labels = labels[areas>minSpheroidArea]
################## filter by distance and get the index
indexOfMinDistance = labels[distancesToCenter == min(distancesToCenter)]
indexOfMinDistance = indexOfMinDistance.astype(int) -1

selectedCC = (labeledImage == allProperties[indexOfMinDistance].label)

spheroidBWImage = selectedCC.astype("uint8")

### actually not neccassary
boundingBox =  allProperties[indexOfMinDistance].bbox

################## get all geometric measurements
centroid = allProperties[indexOfMinDistance].centroid
perimeter = allProperties[indexOfMinDistance].perimeter
area = allProperties[indexOfMinDistance].area
diameter = allProperties[indexOfMinDistance].equivalent_diameter
majorAxis = allProperties[indexOfMinDistance].major_axis_length
minorAxis = allProperties[indexOfMinDistance].minor_axis_length
circularity = 4*math.pi*(area/perimeter**2)

################## Make output images square
### actually not neccassary
outputImageSide = max(boundingBox[2]-boundingBox[0],boundingBox[3]-boundingBox[1])

minRow = round(centroid[0]-outputImageSide/2)
maxRow = round(centroid[0]+outputImageSide/2)
minCol = round(centroid[1]-outputImageSide/2)
maxCol = round(centroid[1]+outputImageSide/2)

croppedBWImage = spheroidBWImage[minRow:maxRow,minCol:maxCol]
croppedImage  =  inputImage[minRow:maxRow,minCol:maxCol]*croppedBWImage

spheroidAttributes = {'area': area ,'diameter': diameter,'circularity': circularity,'majorAxis': majorAxis,'minorAxis': minorAxis}

fullImage = inputImage*spheroidBWImage

plt.figure(1)
plt.imshow(spheroidBWImage)
plt.set_cmap('gray')
plt.axis('on')
plt.figure(2)
plt.imshow(fullImage)
plt.set_cmap('gray')
plt.axis('on')
plt.figure(3)
plt.imshow(croppedBWImage) ### not working, image sizes totally off, issue possibly in squaring part
plt.set_cmap('gray')
plt.axis('on')
plt.figure(4)
plt.imshow(croppedImage) ### not working, image sizes totally off
plt.set_cmap('gray')
plt.axis('on') 

plt.show()

  