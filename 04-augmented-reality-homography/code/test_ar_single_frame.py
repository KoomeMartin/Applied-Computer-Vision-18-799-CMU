"""Test AR on a single frame to debug issues"""
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH


def resize_to_match(img, target_h, target_w):
    """Resize and center crop to match target dimensions"""
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        new_h = target_h
        new_w = int(target_h * current_aspect)
    else:
        new_w = target_w
        new_h = int(target_w / current_aspect)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_y = (new_h - target_h) // 2
    start_x = (new_w - target_w) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    return cropped


# Load data
cover = cv2.imread(os.path.join( "data", "cv_cover.jpg"))
book_frames = loadVid(os.path.join( "data", "book.mov"))
ar_frames = loadVid(os.path.join("data", "ar_source.mov"))

# Test on frame 50
frame_idx = 50
frame = book_frames[frame_idx]
ar_src = ar_frames[frame_idx]

print(f"Cover shape: {cover.shape}")
print(f"Frame shape: {frame.shape}")
print(f"AR source shape: {ar_src.shape}")

# Match features
matches, locs1, locs2 = matchPics(cover, frame)
print(f"\nFound {len(matches)} matches")

if len(matches) >= 4:
    # Extract points
    pts1 = locs1[matches[:, 0], :]
    pts2 = locs2[matches[:, 1], :]
    x1 = pts1[:, [1, 0]].astype(float)
    x2 = pts2[:, [1, 0]].astype(float)
    
    # Compute homography
    H2to1, inliers = computeH_ransac(x1, x2)
    inlier_count = int(np.sum(inliers))
    print(f"Inliers: {inlier_count}/{len(matches)} ({100*inlier_count/len(matches):.1f}%)")
    print(f"\nHomography H2to1:")
    print(H2to1)
    
    # Prepare AR template
    h_cover, w_cover = cover.shape[:2]
    ar_template = resize_to_match(ar_src, h_cover, w_cover)
    print(f"\nAR template shape: {ar_template.shape}")
    print(f"Cover shape: {cover.shape}")
    
    # Create composite
    composite = compositeH(H2to1, ar_template, frame)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(cover, cv2.COLOR_BGR2RGB))
    plt.title('Cover (Reference)')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Book Frame {frame_idx}')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(ar_template, cv2.COLOR_BGR2RGB))
    plt.title('AR Template (Resized)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.title('Composite Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./results/test_single_frame.png', dpi=150, bbox_inches='tight')
    print("\nSaved test result to results/test_single_frame.png")
    plt.show()
else:
    print("Not enough matches!")
