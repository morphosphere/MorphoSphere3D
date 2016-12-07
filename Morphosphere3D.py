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
from ggplot import *
from PIL import Image
#import re
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
        inputImage = inputImage.astype(int) * 255

    plt.imshow(inputImage, 'gray')
    plt.set_cmap('gray')
    plt.axis('on')
    plt.show()

def visualizeSaveMatrix(inputImage, stepName):
    if inputImage.dtype == 'bool':
        inputImage = inputImage.astype(int)*255
    
    plt.imshow(inputImage, 'gray')
    plt.set_cmap('gray')
    plt.axis('on')
    plt.show()
    im = Image.fromarray(inputImage)
    im.save(stepName)

def plotRDF(rdf, fileName):
    plot = ggplot(aes(x='r',y='IntensityNuclei'), data=rdf) + \
        geom_point(color= 'steelblue')
    print plot
    plot.save(fileName[:-4] + '_Nuclei_RDF.png')

    plot = ggplot(aes(x='r',y='IntensityVirus'), data=rdf) + \
        geom_point(color= '#33CC33')
    print plot
    plot.save(fileName[:-4] + '_Virus_RDF.png')

# combining two datasets in one plot does not work yet although claimed otherwise
# alterantively, combine both datasets into one and print as facet_grid
# def plotRDF(rdfNuclei, fileNameNuclei, rdfVirus, fileNameVirus):
#     ##### ##### ##### To do: plot in same graph, axis labels, x axis in um, y min 0
#     rdfPlot = ggplot(aes(x = 'r', y = 'Intensity'), data = rdfNuclei) + \
#               geom_point('steelblue') + \
#               geom_point(x = 'r', y = 'Intensity', data = rdfVirus, color = '#33CC33')
#     print rdfPlot
#     rdfPlot.save(fileNameVirus[:-4] + '_RDF.png')

def label2mask(labeledImage,label):
    labeledImage[labeledImage!=label] = 0
    labeledImage[labeledImage==label] = 1
    return labeledImage
    #for unique_counts(labeledImage)

def computeRDF(inputImageNuclei, inputImageVirus, centroid, diameter, numberOfBins):
    ################## preallocate matrix for r values
    n, m = np.shape(inputImageVirus)
    radiusToCentroid = np.zeros((n, m))

    ################## define DataFrame
    columns = ['r', 'IntensityNuclei', 'IntensityVirus']
    rdf = pd.DataFrame(columns=columns)

    ################## calculate radius bin size and max radius
    radiusDelta = np.floor((diameter/2)/numberOfBins)
    radiusMax = (math.ceil((diameter/2)/radiusDelta)) * radiusDelta  # segmented radius to rounded up n*radiusDelta

    ################## initialize vectors for mean intensity calculations
    radiusValues = np.arange(radiusDelta, radiusMax + radiusDelta, radiusDelta) # calculate RDF for more than diameter to see intensity drop and account for imperfect roundness
    intensityNucleiSum = np.zeros(len(radiusValues))
    intensityNucleiCounter = np.zeros(len(radiusValues))
    intensityNuclei = np.zeros(len(radiusValues))
    intensityVirusSum = np.zeros(len(radiusValues))
    intensityVirusCounter = np.zeros(len(radiusValues))
    intensityVirus = np.zeros(len(radiusValues))

    ################## calculate each pixel's distance to the centroid and add intensity to respective bin's sum
    for iN in range(0, n):
        for iM in range(0, m):
            radiusToCentroid[iN, iM] = math.hypot((iN - centroid[0]), (iM - centroid[1]))
            radiusValue = math.floor(radiusToCentroid[iN, iM] / radiusDelta)
            if radiusValue <= len(radiusValues) - 1:
                intensityNucleiSum[radiusValue] = intensityNucleiSum[radiusValue] + inputImageNuclei[iN, iM]
                intensityNucleiCounter[radiusValue] = intensityNucleiCounter[radiusValue] + 1
                intensityVirusSum[radiusValue] = intensityVirusSum[radiusValue] + inputImageVirus[iN, iM]
                intensityVirusCounter[radiusValue] = intensityVirusCounter[radiusValue] + 1

    ################## calculate mean intensities and append DataFrame
    for iRadius in range(0, len(radiusValues)):
        intensityNuclei[iRadius] = intensityNucleiSum[iRadius] / intensityNucleiCounter[iRadius]
        intensityVirus[iRadius] = intensityVirusSum[iRadius] / intensityVirusCounter[iRadius]
        iRDF = pd.DataFrame({'r': [iRadius * radiusDelta], 'IntensityNuclei' : [intensityNuclei[iRadius]], 'IntensityVirus': [intensityVirus[iRadius]]})  # intensity between iRadius and iRadius + radiusDelta
        rdf = rdf.append(iRDF)

    return rdf.sort('r')
    
def getCentralRegionAndProperties(labeledImage, imageHeight, imageWidth):
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
fileNameNuclei = 'B01_405_z720.tif'
fileNameVirus = 'B01_488_z720.tif'
inputImageNuclei = cv2.imread(fileNameNuclei, -1) #0 = grey, 1=RGB, -1=unchanged
inputImageVirus = cv2.imread(fileNameVirus, -1) #0 = grey, 1=RGB, -1=unchanged

################## validate input visually
#visualizeMatrix(inputImageNuclei)
#visualizeMatrix(inputImageVirus)

################## convert input image to 8bit
processedImageNuclei = processGrayImage(inputImageNuclei)
processedImageVirus = processGrayImage(inputImageVirus)

################## smoothen and threshold input image
gaussianSigma = 5
thresholdedImageNuclei = thresholdImage(processedImageNuclei, gaussianSigma)

################## segment spheroid area
minSpheroidArea = 500
dilationDisk = 100
blockSize = 501

processedBinaryImage = processBinaryImage(thresholdedImageNuclei, dilationDisk)
stepNameBinaryNuclei = fileNameNuclei[:-4] + '_processedBinaryImageNuclei' + '.tif'
#visualizeSaveMatrix(processedBinaryImage, stepNameBinaryNuclei)

################## calculate center of spheroid
imageHeight, imageWidth = inputImageNuclei.shape[:2]
labeledImageNuclei = measure.label(thresholdedImageNuclei)
label, centroid, perimeter, area, diameter, majorAxis, minorAxis, circularity, boundingBox = getCentralRegionAndProperties(labeledImageNuclei, imageHeight, imageWidth)

labeled2mask = label2mask(labeledImageNuclei,label)
maskedImageNuclei = np.multiply(inputImageNuclei, labeled2mask)
maskedImageVirus = np.multiply(inputImageVirus,labeled2mask)

stepNameMaskedNuclei = fileNameNuclei[:-4] + '_maskedImageNuclei' + '.tif'
#visualizeSaveMatrix(maskedImageNuclei, stepNameMaskedNuclei)
stepNameMaskedVirus = fileNameVirus[:-4] + '_maskedImageVirus' + '.tif'
#visualizeSaveMatrix(maskedImageVirus, stepNameMaskedVirus)

#numberOfProcesses = 4
#pool = multiprocessing.Pool(processes=numberOfProcesses)
#maskedImageRDF = pool.map(computeRDF, inputImage, centroid)
#pool.close()
#pool.join()

################## calculate radial distribution function
numberOfBins = 100 # numberOfBins = 'max' for single pixel resolution
maskedImageRDF = computeRDF(inputImageNuclei,inputImageVirus, centroid, diameter, numberOfBins)

################## plot RDFs
plotRDF(maskedImageRDF, fileNameNuclei)

################## calculate radial distribution function
maskedImageRDF.to_csv(fileNameNuclei[:-4] + '_RDF.csv', sep=',')

print 'Ran MorphoSphere3D for ' + fileNameNuclei[:-4]