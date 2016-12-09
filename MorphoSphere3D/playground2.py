"""
Playground2: Development of MorphoSphere3D
@authors: Fanny Georgi

"""

import skimage
import math
import cv2
from scipy import ndimage
from skimage import measure,morphology, data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *
from PIL import Image
import re
#import multiproessing
#import tiffcapture as tc
import skimage.io

im = data.binary_blobs(128, n_dim = 3, volume_fraction = 0.2)
print im.shape
labels = measure.label(im)
print type(im), type(labels)
labels[0,0,0]



print 'done' ################## breakpoint is happy here