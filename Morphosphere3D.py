"""
Morphspere 3D: analysis platform for lightsheet microscopy of organoids
@authors: Fanny Georgi, Vardan Andriasyan, Artur Yakimovich

"""

import skimage
import math
import cv2
from scipy import ndimage
from skimage import measure,morphology
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ggplot as gg
#import multiproessing


def processGrayImage(inputImage):
    processedImage = (inputImage*255).astype('uint8')
    processedImage = np.asarray(processedImage)
    return processedImage
    
def processBinaryImage(inputImage, dilationDisk):
    selem = skimage.morphology.disk(dilationDisk)
    processedImage = cv2.dilate(inputImage,selem,iterations = 1)
    processedImage = ndimage.binary_fill_holes(processedImage)
    return processedImage
    
def thresholdImage(inputImage, gaussianSigma):
    blur = cv2.GaussianBlur(inputImage,(gaussianSigma,gaussianSigma),0)
    threshold, thresholdedImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholdedImage
    
def visualizeMatrix(inputImage):
    if inputImage.dtype == 'bool':
        inputImage = inputImage.astype(int)*255
    
    plt.imshow(inputImage, 'gray')
    plt.set_cmap('gray')
    plt.axis('on')
    plt.show()
    
def plotRDF(rdf):
    print gg.ggplot(rdf, gg.aes('Intensity', 'r')) + \
        gg.geom_line(rdf, colour='steelblue')


def label2mask(labeledImage,label):
    labeledImage[labeledImage!=label] = 0
    labeledImage[labeledImage==label] = 1
    return labeledImage
    #for unique_counts(labeledImage)
    
def cartesian2Polar(x,y):
    r = math.sqrt(np.square(x)+np.square(y))
    phi = math.atan2(x,y)
    return r, phi
    
    
def cartesian2Spheric(x,y,z):
    r = math.sqrt(np.square(x)+np.square(y)+np.square(z))
    theta = math.atan2(y,x)
    phi = math.acos(z/r)
    return r, theta, phi
    
def recenterCertesian(centroid,x,y,z=False):
    if z == False:
        xCentroid,yCentroid = centroid
        x = x-xCentroid
        y= y-yCentroid
        return x, y
    else:
        xCentroid,yCentroid,zCentroid = centroid
        x = x-xCentroid
        y= y-yCentroid
        z=z-zCentroid
        return x, y, z
        
def computeRDF(inputImage, centroid):  
    n,m = np.shape(inputImage)
    #preallocate matrix for r values
    rValues = np.zeros((n,m))
    columns = ['r','Intensity']
    rdf = pd.DataFrame(columns=columns)
    
    for iN in range(0,n):
        vecM = np.arange(m)
        x,y = recenterCertesian(centroid,iN,vecM)
        vecCartesian2Polar = np.vectorize(cartesian2Polar)
        r, phi = vecCartesian2Polar(x, y)
        #r, phi = cartesian2Polar(x,y)
        rValues[iN,vecM] = r
    values = np.unique(rValues)
    for iRvalue in values:
        iIntensity = inputImage[np.where(rValues[rValues == iRvalue])].mean
        iRDF = pd.DataFrame({'r': [iRvalue], 'Intensity': [iIntensity]})
        rdf.append(iRDF)
    return rdf.sort('r')
    
def getCentralRegionAndProperties(inputImage, imageHeight, imageWidth):
    allProperties = measure.regionprops(labeledImage)
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
    
    #selectedCC = (labeledImage == allProperties[indexOfMinDistance].label)
    
    #spheroidBWImage = selectedCC.astype("uint8")
    
    ### actually not neccassary
    boundingBox =  allProperties[indexOfMinDistance].bbox
    
    label = allProperties[indexOfMinDistance].label
    centroid = allProperties[indexOfMinDistance].centroid
    perimeter = allProperties[indexOfMinDistance].perimeter
    area = allProperties[indexOfMinDistance].area
    diameter = allProperties[indexOfMinDistance].equivalent_diameter
    majorAxis = allProperties[indexOfMinDistance].major_axis_length
    minorAxis = allProperties[indexOfMinDistance].minor_axis_length
    circularity = 4*math.pi*(area/perimeter**2)
    
    return label, centroid, perimeter, area, diameter, majorAxis, minorAxis, circularity, boundingBox
    
    

################## open single z plane from 16bit image
fileName = 'B01_405_z720.tif'
inputImage = cv2.imread(fileName, -1) #0 = grey, 1=RGB, -1=unchanged

################## validate input visually
visualizeMatrix(inputImage)
################## convert to 8bit

minSpheroidArea = 500
dilationDisk = 100
blockSize = 501
gaussianSigma = 5
numberOfProcesses = 4

imageHeight, imageWidth = inputImage.shape[:2]

processedImage = processGrayImage(inputImage)
thresholdedImage = thresholdImage(processedImage, gaussianSigma)
processedBinaryImage = processBinaryImage(thresholdedImage, dilationDisk)
visualizeMatrix(processedBinaryImage)

labeledImage = measure.label(thresholdedImage)

getCentralRegionAndProperties(labeledImage, imageHeight, imageWidth)
label, centroid, perimeter, area, diameter, majorAxis, minorAxis, circularity, boundingBox = getCentralRegionAndProperties(labeledImage, imageHeight, imageWidth)

maskedImage = np.multiply(inputImage,label2mask(labeledImage,label))
#pool = multiprocessing.Pool(processes=numberOfProcesses)
#maskedImageRDF = pool.map(computeRDF, inputImage, centroid)
#pool.close()
#pool.join() 
maskedImageRDF = computeRDF(inputImage, centroid)

visualizeMatrix(maskedImage)
plotRDF(maskedImageRDF)

################## Make output images square
### actually not neccassary
#outputImageSide = max(boundingBox[2]-boundingBox[0],boundingBox[3]-boundingBox[1])

#minRow = round(centroid[0]-outputImageSide/2)
#maxRow = round(centroid[0]+outputImageSide/2)
#minCol = round(centroid[1]-outputImageSide/2)
#maxCol = round(centroid[1]+outputImageSide/2)

#croppedBWImage = spheroidBWImage[minRow:maxRow,minCol:maxCol]
#croppedImage  =  inputImage[minRow:maxRow,minCol:maxCol]*croppedBWImage

#spheroidAttributes = {'area': area ,'diameter': diameter,'circularity': circularity,'majorAxis': majorAxis,'minorAxis': minorAxis}

#fullImage = inputImage*spheroidBWImage

#plt.figure(1)
#plt.imshow(spheroidBWImage)
#plt.set_cmap('gray')
#plt.axis('on')
#plt.figure(2)
#plt.imshow(fullImage)
#plt.set_cmap('gray')
#plt.axis('on')
#plt.figure(3)
#plt.imshow(croppedBWImage) ### not working, image sizes totally off, issue possibly in squaring part
#plt.set_cmap('gray')
#plt.axis('on')
#plt.figure(4)
#plt.imshow(croppedImage) ### not working, image sizes totally off
#plt.set_cmap('gray')
#plt.axis('on') 

#plt.show()

  