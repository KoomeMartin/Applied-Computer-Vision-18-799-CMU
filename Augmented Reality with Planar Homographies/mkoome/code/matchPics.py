import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
#Complete functions above this line before this step
def matchPics(I1, I2):
    # I1, I2 : Images to match

    # Convert Images to GrayScale
    I1_gray = skimage.color.rgb2gray(I1) if len(I1.shape) == 3 else I1
    I2_gray = skimage.color.rgb2gray(I2) if len(I2.shape) == 3 else I2

    # Detect Features in Both Images
    sigma = 0.12  # Lower sigma for more features (better for video tracking)
    locs1 = corner_detection(I1_gray, sigma)
    locs2 = corner_detection(I2_gray, sigma)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)

    # Match features using the descriptors
    ratio = 0.7  # Higher ratio for more matches (better for video tracking)
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
