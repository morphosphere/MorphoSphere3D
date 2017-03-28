"""
test_calculateMeanRadius: unit test for MorphoSphere3D function calculateMeanRadius
@authors: Fanny Georgi, Vardan Andriasyan, Artur Yakimovich

"""

import math
import cv2
from scipy import ndimage
from skimage import measure,morphology, io, filters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('running tests...')

################## Function to test ##################
###############################################
'''
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
'''
