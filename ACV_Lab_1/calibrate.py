import numpy as np
import cv2
import glob
import yaml
import os
 
"""
After capturing images of a chessboard pattern, this script calibrates the camera
and saves the calibration parameters to a YAML file.
"""
 
# Chessboard settings
CHESSBOARD_SIZE = (9, 7)  # number of inner corners per chessboard row and column 
SQUARE_SIZE = 16.5  # square size of each square in millimeters (mm)
IMAGE_DIR = "calib_images" # this is the directory where the calibration images are stored
OUTPUT_FILE = "calibration.yaml" # Output calibration file

# Print configuration info
print("=" * 50)
print("CAMERA CALIBRATION - Compute Parameters")
print("=" * 50)
print(f"\nConfiguration:")
print(f"  - Image directory: {IMAGE_DIR}")
print(f"  - Chessboard size: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
print(f"  - Square size: {SQUARE_SIZE} mm")
print(f"  - Output file: {OUTPUT_FILE}")
print("=" * 50)

# Check if image directory exists
if not os.path.exists(IMAGE_DIR):
    print(f"\n✗ ERROR: Directory '{IMAGE_DIR}' not found.")
    print("  Please run the image capture script first.")
    exit(1)
 
# Prepare object points based on the chessboard size and square size
# objp will hold the 3D coordinates of the chessboard corners in the world space
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
 
# Generate the grid points in the chessboard pattern
# objp[:, :2] will hold the x, y coordinates of the corners
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
 
# Scale the points by the size of each square
objp *= SQUARE_SIZE
 
# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points
 
# Load images from the specified directory
images = glob.glob(f"{IMAGE_DIR}/*.jpg")

if len(images) == 0:
    print(f"\n✗ ERROR: No .jpg images found in '{IMAGE_DIR}'.")
    print("  Please capture calibration images first.")
    exit(1)

print(f"\nFound {len(images)} images. Processing...")
print("-" * 50)

successful = 0
failed = 0
gray = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"  ✗ {os.path.basename(fname)}: Could not read image")
        failed += 1
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
 
    # If corners are found, continue
    if found:
        objpoints.append(objp)
        # Refine the corner locations
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        # Append the refined corners to imgpoints list
        imgpoints.append(corners2)
        print(f"  ✓ {os.path.basename(fname)}: Chessboard detected")
        successful += 1
    else:
        print(f"  ✗ {os.path.basename(fname)}: Chessboard NOT detected")
        failed += 1

print("-" * 50)
print(f"Results: {successful} successful, {failed} failed")

# Check if we have enough images for calibration
if successful < 3:
    print(f"\n✗ ERROR: Not enough valid images for calibration.")
    print(f"  Need at least 3 images with detected chessboard, got {successful}.")
    print("  Tips: Ensure good lighting, hold pattern steady, check CHESSBOARD_SIZE.")
    exit(1)

print(f"\nRunning camera calibration with {successful} images...")
 
# Calibrate the camera using the collected object points and image points
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
except cv2.error as e:
    print(f"\n✗ ERROR: Calibration failed - {e}")
    exit(1)

# Evaluate calibration quality
print("\n" + "=" * 50)
print("CALIBRATION RESULTS")
print("=" * 50)

print(f"\nReprojection Error: {ret:.4f} pixels")
if ret < 0.5:
    print("  ✓ Excellent calibration quality")
elif ret < 1.0:
    print("  ✓ Good calibration quality")
elif ret < 2.0:
    print("  ~ Acceptable calibration quality")
else:
    print("  ⚠ Poor calibration - consider recapturing images")

print(f"\nCamera Matrix (Intrinsic Parameters):")
print(f"  Focal length (fx): {mtx[0, 0]:.2f} px")
print(f"  Focal length (fy): {mtx[1, 1]:.2f} px")
print(f"  Principal point (cx): {mtx[0, 2]:.2f} px")
print(f"  Principal point (cy): {mtx[1, 2]:.2f} px")

print(f"\nDistortion Coefficients:")
print(f"  k1: {dist[0, 0]:.6f}")
print(f"  k2: {dist[0, 1]:.6f}")
print(f"  p1: {dist[0, 2]:.6f}")
print(f"  p2: {dist[0, 3]:.6f}")
print(f"  k3: {dist[0, 4]:.6f}")

calib_data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "reprojection_error": float(ret)
}
 
# Save the calibration data to a YAML file
with open(OUTPUT_FILE, "w") as f:
    yaml.dump(calib_data, f)
 
print("=" * 50)
print(f"✓ Calibration saved to '{OUTPUT_FILE}'")
print("=" * 50)