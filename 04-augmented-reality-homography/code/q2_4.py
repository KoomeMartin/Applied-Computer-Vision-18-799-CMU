import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
import os

## Make sure you can read files from the data folder
cv_cover_path = os.path.join('data', 'cv_cover.jpg')
cv_desk_path = os.path.join('data', 'cv_desk.png')

cv_cover = cv2.imread(cv_cover_path)
cv_desk = cv2.imread(cv_desk_path)

if cv_cover is None:
    raise FileNotFoundError(f"Image not found at {cv_cover_path}")
if cv_desk is None:
    raise FileNotFoundError(f"Image not found at {cv_desk_path}")

matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

# display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
