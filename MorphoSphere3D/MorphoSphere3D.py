"""
MorphoSphere3D: analysis platform for lightsheet microscopy of organoids and spheroids
@authors: Fanny Georgi, Vardan Andriasyan, Artur Yakimovich

"""

import math
import cv2
from scipy import ndimage
from skimage import measure,morphology, io, filters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *

################## Functions ##################
###############################################

def calculateMeanRadius(labeled2mask, centroid, lenX, lenY, lenZ):
    # shift image for boundary condition
    imageWithBoundary = np.zeros((lenZ + 2, lenY + 2, lenX + 2))
    for iX in range(0, lenX, 1):
        for iY in range(0, lenY, 1):
            for iZ in range(0, lenZ, 1):
                imageWithBoundary[iZ + 1, iY + 1, iX + 1] = labeled2mask[iZ, iY, iX]

    # identify contour
    contourWithBoundary = np.zeros((lenZ + 2, lenY + 2, lenX + 2))
    for iX in range(0, lenX + 2, 1):
        for iY in range(0, lenY + 2, 1):
            for iZ in range(0, lenZ + 2, 1):
                if imageWithBoundary[iZ, iY, iX] == 1:
                    if imageWithBoundary[(iZ - 1), (iY), (iX)] != 1 or imageWithBoundary[(iZ), (iY), (iX - 1)] != 1 or imageWithBoundary[(iZ), (iY - 1), (iX)] != 1 or imageWithBoundary[(iZ), (iY + 1), (iX)] != 1 or imageWithBoundary[(iZ), (iY), (iX + 1)] != 1 or imageWithBoundary[(iZ + 1), (iY), (iX)] != 1:
                        contourWithBoundary[iZ, iY, iX] = 1

    # shift image back
    contour = np.zeros((lenZ, lenY, lenX))
    for iX in range(0, lenX, 1):
        for iY in range(0, lenY, 1):
            for iZ in range(0, lenZ, 1):
                contour[iZ, iY, iX] = contourWithBoundary[iZ + 1, iY + 1, iX + 1]

    # plt.imshow(contour[70, :, :])

    # calculate mean radius
    radiusSum = 0
    radiusCounter = 0
    for iX in range(0, lenX, 1):
        for iY in range(0, lenY, 1):
            for iZ in range(0, lenZ, 1):
                if contour[iZ, iY, iX] == 1:
                    radiusSum = radiusSum + math.sqrt((iX - centroid[2]) ** 2 + (iY - centroid[1]) ** 2 + (iZ - centroid[0]) ** 2)
                    radiusCounter = radiusCounter + 1

    meanRadius = radiusSum / radiusCounter
    return meanRadius

################## Version A: Calculating each pixel's distance to the centroid as hypotenuse and store intensity information in vectors
def computeRDF(inputImageNuclei, inputImageVirus, centroid, diameter, numberOfBins):
    # preallocate matrix for r values
    radiusToCentroid = np.zeros((lenZ, lenY, lenX))

    # define DataFrame
    columns = ['r', 'IntensityNuclei', 'IntensityVirus']
    rdf = pd.DataFrame(columns=columns)

    radiusDelta = np.floor((diameter / 2) / numberOfBins)
    RadiusMax = (math.ceil((diameter / 2) / radiusDelta)) * radiusDelta  # segmented radius to full n*radiusDelta

    radiusValues = np.arange(radiusDelta, RadiusMax + radiusDelta, radiusDelta)
    intensityNucleiSum = np.zeros(len(radiusValues))
    intensityNucleiCounter = np.zeros(len(radiusValues))
    intensityNuclei = np.zeros(len(radiusValues))
    intensityVirusSum = np.zeros(len(radiusValues))
    intensityVirusCounter = np.zeros(len(radiusValues))
    intensityVirus = np.zeros(len(radiusValues))

    # sum intensities in bins
    for iX in range(0, lenX, 1):
        for iY in range(0, lenY, 1):
            for iZ in range(0, lenZ, 1):
                radiusToCentroid[iZ, iY, iX] = math.sqrt(
                    (iX - centroid[2]) ** 2 + (iY - centroid[1]) ** 2 + (iZ - centroid[0]) ** 2)
                radiusValue = math.floor(radiusToCentroid[iZ, iY, iX] / radiusDelta)
                if radiusValue <= len(radiusValues) - 1:
                    intensityNucleiSum[radiusValue] = intensityNucleiSum[radiusValue] + inputImageNuclei[iZ, iY, iX]
                    intensityNucleiCounter[radiusValue] = intensityNucleiCounter[radiusValue] + 1
                    intensityVirusSum[radiusValue] = intensityVirusSum[radiusValue] + inputImageVirus[iZ, iY, iX]
                    intensityVirusCounter[radiusValue] = intensityVirusCounter[radiusValue] + 1

    # average intensities
    for iRadius in range(0, len(intensityNuclei)):
        intensityNuclei[iRadius] = intensityNucleiSum[iRadius] / intensityNucleiCounter[iRadius]
        intensityVirus[iRadius] = intensityVirusSum[iRadius] / intensityVirusCounter[iRadius]
        iRDF = pd.DataFrame({'r': [iRadius * radiusDelta], 'IntensityNuclei': [intensityNuclei[iRadius]], 'IntensityVirus': [intensityVirus[iRadius]]})  # intensity between iRadius and iRadius + radiusDelta
        rdf = rdf.append(iRDF)
    return rdf.sort('r')
################## End of version A

def getCentralRegionAndProperties(labeledImage, lenZ, lenY, lenX, minSpheroidVolume):
    allProperties = measure.regionprops(labeledImage)
    ################## initialize empty arrays for area and distance filtering
    areas = np.empty(len(allProperties))
    distancesToCenter = np.empty(len(allProperties))
    labels = np.empty(len(allProperties))

    ################## find the index connected area which is closest to center of the image also area filter
    i = 0
    for region in allProperties:
        z0, y0, x0 = ndimage.measurements.center_of_mass(region.image)
        distance = math.sqrt((z0 - lenZ) ** 2 + (y0 - lenY / 2) ** 2 + (x0 - lenX / 2) ** 2)
        distancesToCenter[i] = distance
        areas[i] = region.area
        labels[i] = region.label
        i = i + 1

    ################## filter by area
    distancesToCenter = distancesToCenter[areas > minSpheroidVolume]
    labels = labels[areas > minSpheroidVolume]
    ################## filter by distance and get the index
    indexOfMinDistance = labels[distancesToCenter == min(distancesToCenter)]
    indexOfMinDistance = indexOfMinDistance.astype(int) - 1

    # selectedCC = (labeledImage == allProperties[indexOfMinDistance].label
    # spheroidBWImage = selectedCC.astype("uint8")

    ### actually not neccassary
    boundingBox = allProperties[indexOfMinDistance].bbox

    label = allProperties[indexOfMinDistance].label
    centroid = ndimage.measurements.center_of_mass(allProperties[indexOfMinDistance].image)
    #perimeter = allProperties[indexOfMinDistance].perimeter
    area = allProperties[indexOfMinDistance].area
    #diameter = allProperties[indexOfMinDistance].equivalent_diameter
    #majorAxis = allProperties[indexOfMinDistance].major_axis_length
    #minorAxis = allProperties[indexOfMinDistance].minor_axis_length
    #circularity = 4 * math.pi * (area / perimeter ** 2)
    #sphericity = (2 * (majorAxis * minorAxis **2) **(1/3)) / (majorAxis + (minorAxis**2 / (majorAxis**2 - minorAxis**2)**(1/2)) * log((majorAxis + (majorAxis**2 - minorAxis**2)**(1/2)) / minorAxis)) #### ! 1/2

    return label, centroid, area, boundingBox

def label2mask(labeledImage,label):
    labeledImage[labeledImage!=label] = 0
    labeledImage[labeledImage==label] = 1
    return labeledImage

def plotRDF(rdf, fileName):
    plot = ggplot(aes(x='r',y='IntensityNuclei'), data=rdf) + \
        geom_point(color= 'steelblue')
    print plot
    plot.save(fileName[:-4] + '_Nuclei_RDF.png')

    plot = ggplot(aes(x='r',y='IntensityVirus'), data=rdf) + \
        geom_point(color= '#33CC33')
    print plot
    plot.save(fileName[:-4] + '_Virus_RDF.png')

################## combining two datasets in one plot does not work yet although claimed otherwise
################## alterantively, combine both datasets into one and print as facet_grid
# def plotRDF(rdfNuclei, fileNameNuclei, rdfVirus, fileNameVirus):
#     ################## To do: plot in same graph, axis labels, x axis in um, y min 0
#     rdfPlot = ggplot(aes(x = 'r', y = 'Intensity'), data = rdfNuclei) + \
#               geom_point('steelblue') + \
#               geom_point(x = 'r', y = 'Intensity', data = rdfVirus, color = '#33CC33')
#     print rdfPlot
#     rdfPlot.save(fileNameVirus[:-4] + '_RDF.png')

def processBinary(processedImage, sigma):
    ################## gaussian
    gaussian = filters.gaussian(processedImage, sigma=sigma)
    ################## threshold binary otsu
    threshold = filters.threshold_otsu(gaussian)
    binary = gaussian > threshold
    ################## dilate and fill holes
    dilated = morphology.binary_dilation(binary)  # dilation disk?
    processedBinary = ndimage.binary_fill_holes(dilated)
    return processedBinary

def processGrayImage(inputImage):
    processedImage = cv2.convertScaleAbs(inputImage, alpha=(255.0/65535.0))
    processedImage = np.asarray(processedImage)
    return processedImage

def visualizeMatrix(inputImage):
    if inputImage.dtype == 'bool':
        inputImage = inputImage.astype(int) * 255
    ################## convert to 8bit
    if inputImage.dtype == 'int64':
        inputImage = cv2.convertScaleAbs(inputImage, alpha=(255.0/65535.0))

    plt.imshow(inputImage[math.floor(lenZ/2)], 'gray')
    plt.set_cmap('gray')
    plt.axis('on')
    plt.show()

def visualizeSaveMatrix(inputImage, stepName):
    if inputImage.dtype == 'bool':
        inputImage = inputImage.astype(int) * 255 # result is 32bit
    ################## convert to 8bit
    if inputImage.dtype == 'int64':
        inputImage = cv2.convertScaleAbs(inputImage, alpha=(255.0/65535.0))

    plt.imshow(inputImage[math.floor(lenZ/2)], 'gray')
    plt.set_cmap('gray')
    plt.axis('on')
    plt.show()
    io.imsave(stepName, inputImage, plugin='tifffile')

################## Code ##################
##########################################

################## Define input images (16bit, otherwise skip processGray)
fileNameNuclei = 'B01_405_down_8bit.tif'
fileNameVirus = 'B01_488_down_8bit.tif'

################## Read images
inputImageNuclei = io.imread(fileNameNuclei, plugin='tifffile') # dimesions z, y, x
inputImageVirus = io.imread(fileNameVirus, plugin='tifffile') # dimesions z, y, x

################## Stack properties
lenX = len(inputImageNuclei[1,1,:])
lenY = len(inputImageNuclei[1,:,1])
lenZ = len(inputImageNuclei[:,1,1])

################## Convert input image to 8bit
# processedImageNuclei = processGrayImage(inputImageNuclei)
# processedImageVirus = processGrayImage(inputImageVirus)

processedImageNuclei = inputImageNuclei
processedImageVirus = inputImageVirus

################## Show images
# visualizeMatrix(processedImageNuclei)
# visualizeMatrix(processedImageVirus)

################## Segment spheroid in nuclei channel
sigma = 5
processedBinaryImage = processBinary(processedImageNuclei, sigma)
# visualizeMatrix(processedBinaryImage)

################## Label and measure objects
labeledImageNuclei = measure.label(processedBinaryImage)

################## Calculate centroid, segment and get other properties
minSpheroidVolume = 1000000 # sum over all pixels within labeled region
label, centroid, volume, boundingBox = getCentralRegionAndProperties(labeledImageNuclei, lenZ, lenY, lenX, minSpheroidVolume)

################## Create mask of segmented ROI
labeled2mask = label2mask(labeledImageNuclei,label)

################## Calculate mean radius
meanRadius = calculateMeanRadius(labeled2mask, centroid, lenX, lenY, lenZ)
diameter = 2 * meanRadius

################## Apply mask to stacks
maskedImageNuclei = np.multiply(inputImageNuclei, labeled2mask)
maskedImageVirus = np.multiply(inputImageVirus, labeled2mask)

################## Export and visualize central z plane, alternatively perform visulaizeMatrix only
stepNameBinaryNuclei = fileNameNuclei[:-4] + '_processedBinaryImageNuclei' + '.tif'
visualizeSaveMatrix(processedBinaryImage, stepNameBinaryNuclei)

stepNameMaskedImageNuclei = fileNameNuclei[:-4] + '_maskedNuclei' + '.tif'
visualizeSaveMatrix(maskedImageNuclei, stepNameMaskedImageNuclei)

stepNameMaskedImageVirus = fileNameNuclei[:-4] + '_maskedVirus' + '.tif'
visualizeSaveMatrix(maskedImageVirus, stepNameMaskedImageVirus)

################## Calculate radial distribution function
numberOfBins = 10 # numberOfBins = 'max' for single pixel resolution
maskedImageRDF = computeRDF(inputImageNuclei,inputImageVirus, centroid, diameter, numberOfBins)

################## Plot RDFs
plotRDF(maskedImageRDF, fileNameNuclei)

################## Export RDF to csv
maskedImageRDF.to_csv(fileNameNuclei[:-4] + '_RDF.csv', sep=',')

################## Final output
print 'Ran MorphoSphere3D for ' + fileNameNuclei[:-4] + ' with diameter of ' + str(diameter) + ' pixels'