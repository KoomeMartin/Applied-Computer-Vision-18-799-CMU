import numpy as np
import cv2
import skimage.io
import skimage.color
import skimage.transform
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
import os
from matplotlib import pyplot as plt


# Write script for Q2.9

# 1. Read the three images
cv_cover_path = os.path.join("data", "cv_cover.jpg")
cv_desk_path = os.path.join("data", "cv_desk.png")
hp_cover_path = os.path.join("data", "hp_cover.jpg")

cv_cover = cv2.imread(cv_cover_path)
cv_desk = cv2.imread(cv_desk_path)
hp_cover = cv2.imread(hp_cover_path)

if cv_cover is None or cv_desk is None or hp_cover is None:
    raise FileNotFoundError("One or more images not found")

print("Images loaded successfully")
print(f"CV Cover shape: {cv_cover.shape}")
print(f"CV Desk shape: {cv_desk.shape}")
print(f"HP Cover shape: {hp_cover.shape}")

# 2. Compute homography automatically using matchPics and computeH_ransac
print("\nMatching features between cv_cover and cv_desk...")
matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
print(f"Found {len(matches)} matches")

# Extract matched points
# computeH_ransac(x1, x2) computes H such that: x1 = H * x2
# We want: cv_cover_points = H * cv_desk_points
pts1 = locs1[matches[:, 0], :]  # Points in cv_cover (row, col)
pts2 = locs2[matches[:, 1], :]  # Points in cv_desk (row, col)
# Convert to (x, y) order for homography estimation
x1 = pts1[:, [1, 0]].astype(float)
x2 = pts2[:, [1, 0]].astype(float)

# 3. Compute homography using RANSAC
# H2to1 will satisfy: x1 = H2to1 * x2, i.e., cv_cover = H2to1 * cv_desk
print("\nComputing homography with RANSAC...")
H2to1, inliers = computeH_ransac(x1, x2)
inlier_count = int(np.sum(inliers)) if inliers is not None else 0
print(f"Found {inlier_count} inliers out of {len(matches)} matches")

# 4. Resize hp_cover to match cv_cover dimensions robustly (pad to preserve content)
h_cv, w_cv = cv_cover.shape[:2]
hp_h, hp_w = hp_cover.shape[:2]
pad_top = pad_bottom = pad_left = pad_right = 0
target_ar = w_cv / h_cv
curr_ar = hp_w / hp_h
crop_or_pad = hp_cover
if abs(curr_ar - target_ar) > 1e-6:
    if curr_ar > target_ar:
        # Too wide; pad height
        new_h = int(round(hp_w / target_ar))
        diff = max(new_h - hp_h, 0)
        pad_top = diff // 2
        pad_bottom = diff - pad_top
    else:
        # Too tall; pad width
        new_w = int(round(target_ar * hp_h))
        diff = max(new_w - hp_w, 0)
        pad_left = diff // 2
        pad_right = diff - pad_left
hp_cover_padded = cv2.copyMakeBorder(
    hp_cover, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
)
hp_cover_resized = cv2.resize(hp_cover_padded, (w_cv, h_cv))
print(
    f"HP cover padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}"
)
print(f"Resized HP cover to: {(hp_cover_resized.shape[1], hp_cover_resized.shape[0])}")
print(
    f"HP cover resized dimensions: {hp_cover_resized.shape[1]}x{hp_cover_resized.shape[0]}"
)

# 5. Warp hp_cover to the dimensions of cv_desk using the homography
print("\nWarping HP cover to desk dimensions...")
h_desk, w_desk = cv_desk.shape[:2]

# Guard against invalid homography inliers
if H2to1 is None or inliers is None or int(np.sum(inliers)) < 4:
    print("Not enough inliers; skipping warp.")
    warped_hp = None
    composite_img = cv_desk.copy()
else:
    # H2to1 maps: cv_desk -> cv_cover; warp using inverse
    H_inv = np.linalg.inv(H2to1)
    warped_hp = cv2.warpPerspective(hp_cover_resized, H_inv, (w_desk, h_desk))
    print("HP cover warped successfully")
    print(f"Warped image shape: {warped_hp.shape}")
    

print("HP cover warped successfully")
print(f"Warped image shape: {warped_hp.shape}")

# Display the warped image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv_desk, cv2.COLOR_BGR2RGB))
plt.title('CV Desk (Original)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_hp, cv2.COLOR_BGR2RGB))
plt.title('HP Cover (Warped)')
plt.axis('off')
plt.tight_layout()
plt.show()

# 6. Use compositeH to blend warped image with desk image
print("\nCreating composite image using compositeH...")
if H2to1 is None or inliers is None or int(np.sum(inliers)) < 4:
    composite_img = cv_desk.copy()
else:
    composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)

print("Composite image created successfully!")
print(f"Composite image size: {composite_img.shape}")

# Display final composite
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
plt.title("Composite Image: HP Cover on Desk")
plt.axis("off")
plt.tight_layout()
plt.show()

print("\nHarry Potterize complete!")
