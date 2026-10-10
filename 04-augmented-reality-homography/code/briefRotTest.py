import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from scipy import ndimage
from matplotlib import pyplot as plt
import os


#Q3.5
#Read the image and convert to grayscale, if necessary, you can use OpenCV

# Load the original image
img_path = os.path.join('data', 'cv_cover.jpg')
img = cv2.imread(img_path)

if img is None:
	raise FileNotFoundError(f"Image not found at {img_path}")

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# Store match counts
match_counts = []
angles = []

# Angles to visualize
visualization_angles = [0, 10, 90, 180, 350]

for i in range(36):
	#Rotate Image
	angle = i * 10  # 0 to 350 degrees in 10 degree increments
	img_rotated = ndimage.rotate(img, angle, reshape=False)
	
	# Convert rotated image back to uint8 if necessary
	img_rotated = np.uint8(img_rotated * 255) if img_rotated.max() <= 1 else np.uint8(img_rotated)
	
	#Compute features, descriptors and Match features
	try:
		matches, locs1, locs2 = matchPics(img, img_rotated)
		num_matches = len(matches)
	except:
		num_matches = 0
	
	#Update histogram
	match_counts.append(num_matches)
	angles.append(angle)
	print(f"Angle: {angle}°, Matches: {num_matches}")
	
	# Visualize matches at specific angles
	if angle in visualization_angles:
		print(f"\n--- Visualizing matches at {angle}° rotation (Matches: {num_matches}) ---")
		plotMatches(img, img_rotated, matches, locs1, locs2)


#Display histogram
plt.figure(figsize=(12, 6))
plt.bar(angles, match_counts, width=8, edgecolor='black')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('BRIEF Descriptor Robustness to Rotation')
plt.xticks(angles)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

