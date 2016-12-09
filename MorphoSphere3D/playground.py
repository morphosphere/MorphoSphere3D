"""
Playground: Development of MorphoSphere3D
@authors: Fanny Georgi

"""

import skimage
import math
import cv2
from scipy import ndimage
from skimage import measure,morphology
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *
from PIL import Image
import re
#import multiproessing
#import tiffcapture as tc
import skimage.io

fileNameNuclei = 'B01_405_downsampled_16bit.tif'
################## only reads first plane
#inputImageNuclei = cv2.imread(fileNameNuclei, -1) #0 = grey, 1=RGB, -1=unchanged
inputImageNuclei = skimage.io.imread(fileNameNuclei, plugin='tifffile') # dimesions z, y, x

lenX = len(inputImageNuclei[1,1,:])
lenY = len(inputImageNuclei[1,:,1])
lenZ = len(inputImageNuclei[:,1,1])

################## convert input image to 8bit
################## works on stack
def processGrayImage(inputImage):
    processedImage = (inputImage*255).astype('uint8')
    processedImage = np.asarray(processedImage)
    return processedImage

processedImageNuclei = processGrayImage(inputImageNuclei)

################## redundant
# processedImageNuclei = (inputImageNuclei).astype('uint8')
# processedImageNuclei = np.asarray(processedImageNuclei)

################## also doesn't help
#processedImageNuclei = cv2.cvtColor(inputImageNuclei, cv2.COLOR_BGR2GRAY)

################## smoothen and threshold input image
# alternatively, try morphology.ball for 3D dilation
def thresholdImage(inputImage, gaussianSigma):
    blur = cv2.GaussianBlur(inputImage,(gaussianSigma,gaussianSigma),0)
    threshold, thresholdedImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholdedImage

def processBinaryImage(inputImage, dilationDisk):
    selem = skimage.morphology.disk(dilationDisk)
    processedImage = cv2.dilate(inputImage,selem,iterations = 1)
    processedImage = ndimage.binary_fill_holes(processedImage)
    return processedImage

gaussianSigma = 5
thresholdedImageNuclei = np.zeros_like(inputImageNuclei)
dilationDisk = 3
processedBinaryImage = np.zeros_like(inputImageNuclei)

################## for zPlane, image in enumerate(processedImageNuclei):
for zPlane in range(0,lenZ,1):
    thresholdedImageNuclei[zPlane, :, :] = thresholdImage(processedImageNuclei[zPlane, :, :], gaussianSigma)
    processedBinaryImage[zPlane, :, :] = processBinaryImage(thresholdedImageNuclei[zPlane, :, :], dilationDisk)
################## To do: needs to be refined for blur, dilation, etc in z


################## calculate center of spheroid
stackWidth = lenX
stackHeight = lenY
stackDepth = lenZ

labeledImageNuclei = measure.label(processedBinaryImage)

def label2mask(labeledImage,label):
    labeledImage[labeledImage!=label] = 0
    labeledImage[labeledImage==label] = 1
    return labeledImage

labeledImageNuclei = measure.label(processedBinaryImage) ################## connectivity= optional

################################################################################################################## status bar
def getCentralRegionAndProperties(labeledImage, imageHeight, imageWidth):
    allProperties = measure.regionprops(labeledImage)
    ################## initialize empty arrays for area and distance filtering
    areas = np.empty(len(allProperties))
    distancesToCenter = np.empty(len(allProperties))
    labels = np.empty(len(allProperties))

    ################## find the index connected area which is closest to center of the image also area filter
    i = 0
    for props in allProperties:
        y0, x0 = props.centroid
        distance = math.sqrt((y0 - imageHeight / 2) ** 2 + (x0 - imageWidth / 2) ** 2)
        distancesToCenter[i] = distance
        areas[i] = props.area
        labels[i] = props.label
        i = i + 1

    ################## filter by area
    distancesToCenter = distancesToCenter[areas > minSpheroidArea]
    labels = labels[areas > minSpheroidArea]
    ################## filter by distance and get the index
    indexOfMinDistance = labels[distancesToCenter == min(distancesToCenter)]
    indexOfMinDistance = indexOfMinDistance.astype(int) - 1

    # selectedCC = (labeledImage == allProperties[indexOfMinDistance].label)

    # spheroidBWImage = selectedCC.astype("uint8")

    ### actually not neccassary
    boundingBox = allProperties[indexOfMinDistance].bbox

    label = allProperties[indexOfMinDistance].label
    centroid = allProperties[indexOfMinDistance].centroid
    perimeter = allProperties[indexOfMinDistance].perimeter
    area = allProperties[indexOfMinDistance].area
    diameter = allProperties[indexOfMinDistance].equivalent_diameter
    majorAxis = allProperties[indexOfMinDistance].major_axis_length
    minorAxis = allProperties[indexOfMinDistance].minor_axis_length
    circularity = 4 * math.pi * (area / perimeter ** 2)

    return label, centroid, perimeter, area, diameter, majorAxis, minorAxis, circularity, boundingBox

label, centroid, perimeter, area, diameter, majorAxis, minorAxis, circularity, boundingBox = getCentralRegionAndProperties(labeledImageNuclei, imageHeight, imageWidth)

# numberofBins = 5
# centroid = (695.31711103168061, 733.26415925653055)
# diameter = 1300 #934.196417969
#
# # preallocate matrix for r values
# n,m = np.shape(inputImageVirus)
# radiusToCentroid = np.zeros((n, m))
#
# # define DataFrame
# columns = ['r', 'IntensityVirus']
# rdf = pd.DataFrame(columns=columns)
#
# radiusDelta = 20 # aka np.floor((diameter/2)/numberOfBins)
# RadiusMax = (math.ceil((diameter/2)/radiusDelta))*radiusDelta # segmented radius to full n*radiusDelta
#
# radiusValues = np.arange(radiusDelta,RadiusMax+radiusDelta,radiusDelta)
# intensityVirusSum = np.zeros(len(radiusValues))
# intensityVirusCounter = np.zeros(len(radiusValues))
# intensityVirus = np.zeros(len(radiusValues))
#
# for iN in range(0,n):
#     for iM in range(0,m):
#         radiusToCentroid[iN,iM] = math.hypot((iN - centroid[0]), (iM - centroid[1]))
#         radiusValue = math.floor(radiusToCentroid[iN, iM]/radiusDelta)
#         if radiusValue <= len(radiusValues)-1:
#             intensityVirusSum[radiusValue] = intensityVirusSum[radiusValue] + inputImageVirus[iN,iM]
#             intensityVirusCounter[radiusValue] = intensityVirusCounter[radiusValue] + 1
#
# for iRadius in range(0,len(intensityVirus)):
#     intensityVirus[iRadius] = intensityVirusSum[iRadius] / intensityVirusCounter[iRadius]
#     iRDF = pd.DataFrame({'r': [iRadius*radiusDelta], 'IntensityVirus': [intensityVirus[iRadius]]}) # intensity between iRadius and iRadius + radiusDelta
#     rdf = rdf.append(iRDF)

print 'done' ################## breakpoint


############## independant gaussian
#gaussianImageNuclei = ndimage.filters.gaussian_filter(processedImageNuclei, gaussianSigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)