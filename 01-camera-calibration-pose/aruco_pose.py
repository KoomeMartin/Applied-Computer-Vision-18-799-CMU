import numpy as np
import cv2
import yaml
import os

"""
ArUco Marker Pose Estimation
Get full 6DOF pose (position + orientation) of ArUco markers relative to the camera.

This gives you:
- Translation vector (tvec): X, Y, Z position in meters
- Rotation vector (rvec): Rotation in axis-angle format (can convert to Euler, quaternion, etc.)

Use cases:
- Robot localization
- Augmented reality
- Object tracking
- Camera pose estimation (if marker position is known)
"""

# === Configuration ===
CALIBRATION_FILE = "calibration.yaml"  # Path to camera calibration file
MARKER_SIZE = 0.094  # Size of ArUco marker in METERS (9.4cm)
ARUCO_DICT = cv2.aruco.DICT_4X4_250  # ArUco dictionary type

print("=" * 60)
print("ArUco POSE ESTIMATION")
print("=" * 60)

# === Load camera calibration parameters ===
if not os.path.exists(CALIBRATION_FILE):
    print(f"\n✗ ERROR: '{CALIBRATION_FILE}' not found.")
    exit(1)

with open(CALIBRATION_FILE, "r") as f:
    calib_data = yaml.safe_load(f)

camera_matrix = np.array(calib_data["camera_matrix"])
dist_coeffs = np.array(calib_data["dist_coeff"])

print(f"\n✓ Loaded calibration")
print(f"\nConfiguration:")
print(f"  - Marker size: {MARKER_SIZE * 100:.1f} cm")
print(f"\nPose Information Displayed:")
print(f"  - Position (X, Y, Z) in meters")
print(f"  - Rotation (Roll, Pitch, Yaw) in degrees")
print(f"\nCoordinate System:")
print(f"  - X: Right (red axis)")
print(f"  - Y: Down (green axis)")  
print(f"  - Z: Forward/Into screen (blue axis)")
print(f"\nControls:")
print(f"  [ESC] / [Q] - Quit")
print(f"  [P]         - Print detailed pose to terminal")
print("=" * 60)

def rotation_vector_to_euler(rvec):
    """Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees"""
    R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
    
    # Extract Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees([roll, pitch, yaw])

# === Initialize ArUco detector ===
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)  # Get the ArUco dictionary
aruco_params = cv2.aruco.DetectorParameters()  # Default detection parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  # Create detector object

# Marker 3D points in marker coordinate system (centered at (0,0,0))
half_size = MARKER_SIZE / 2

# defines the 3D coordinates of the four corners of the ArUco marker in its own coordinate system
marker_points = np.array([
    [-half_size,  half_size, 0],  # Top-left
    [ half_size,  half_size, 0],  # Top-right
    [ half_size, -half_size, 0],  # Bottom-right
    [-half_size, -half_size, 0],  # Bottom-left
], dtype=np.float32)

# === Open webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ ERROR: Could not open webcam")
    exit(1)

print_detailed = False  # Flag to print detailed pose info

# === Main loop: process each video frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # --- Detect ArUco markers in the frame ---
    corners, ids, rejected = detector.detectMarkers(frame)
    
    display = frame.copy()
    
    if ids is not None and len(ids) > 0:
        # Draw detected marker borders and IDs
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        
        for i, (corner, marker_id) in enumerate(zip(corners, ids)):
            corner_points = corner[0].astype(np.float32)
            
            # --- Estimate pose of the marker ---
            # SolvePnP finds the rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(
                marker_points, corner_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Draw 3D coordinate axes on the marker
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.7)
                
                # Get position and orientation
                x, y, z = tvec.flatten()
                roll, pitch, yaw = rotation_vector_to_euler(rvec)
                distance = np.linalg.norm(tvec)
                
                # Display pose info on the frame
                center = corner_points.mean(axis=0).astype(int)
                
                info_lines = [
                    f"ID: {marker_id[0]} | Dist: {distance:.2f}m",
                    f"Pos: X={x:.2f} Y={y:.2f} Z={z:.2f}",
                    f"Rot: R={roll:.1f} P={pitch:.1f} Y={yaw:.1f}"
                ]
                
                y_offset = -70
                for line in info_lines:
                    cv2.putText(display, line, (center[0] - 80, center[1] + y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    y_offset += 20
                
                # Optionally print detailed pose info to terminal
                if print_detailed:
                    print(f"\n--- Marker {marker_id[0]} ---")
                    print(f"Position (m):  X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
                    print(f"Distance (m):  {distance:.4f}")
                    print(f"Rotation (°):  Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")
                    print(f"Rotation vec:  {rvec.flatten()}")
                    print_detailed = False
    else:
        # No marker detected
        cv2.putText(display, "No ArUco marker detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw axis legend on the frame
    legend_y = 30
    cv2.putText(display, "Axes: ", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(display, "X", (60, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.putText(display, "Y", (80, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(display, "Z", (100, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # Show the result
    cv2.imshow("ArUco Pose Estimation", display)
    
    # --- Keyboard controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('p'):
        print_detailed = True

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("\n✓ Done!")
