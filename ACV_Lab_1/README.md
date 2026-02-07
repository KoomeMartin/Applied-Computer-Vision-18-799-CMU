# ACV Lab 1 - Camera Calibration & ArUco Pose Estimation

This folder contains tools and scripts for camera calibration and ArUco marker pose detection using OpenCV.

## What's Inside

### Core Scripts
- **capture.py** - Captures calibration images from webcam using a chessboard pattern (9x7 inner corners)
- **calibrate.py** - Computes camera intrinsic parameters and distortion coefficients from captured images
- **aruco_pose.py** - Real-time 6DOF pose estimation (position + orientation) of ArUco markers using calibrated camera
- **calibration.yaml** - Saved camera calibration parameters (camera matrix, distortion coefficients)

### Educational Materials
- **camera_calibration_class.ipynb** - Complete tutorial notebook covering camera calibration theory, OpenCV basics, and ArUco pose detection
- **opencv_primer_scripts/** - Beginner-friendly OpenCV examples (reading images/video, drawing shapes, webcam access, image processing)
- **calib_images/** - Sample chessboard calibration images
- **notebook_images/** - Diagrams explaining perspective projection and camera models

## Purpose
Learn and implement camera calibration for accurate 3D computer vision applications including robotics, augmented reality, and measurement systems.
