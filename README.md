# Applied Computer Vision - CMU 18-799

Course materials and assignments for Applied Computer Vision at Carnegie Mellon University. This repository documents the progression from camera fundamentals through advanced computer vision techniques.

## Learning Path

### 🔧 Lab 1: Camera Fundamentals
**ACV_Lab_1** - Foundation for all computer vision work
- Camera calibration workflow (capture → calibrate → use)
- Real-time 6DOF pose tracking with ArUco markers
- OpenCV primer scripts for beginners
- Complete tutorial notebook with theory and practice

### 📐 Assignment 1: Image Formation
**Perspective and Orthographic Projections** - Understanding 3D-to-2D projection
- Perspective and orthographic projection implementations
- Camera orbit animations (120 frames)
- Multi-view orthographic projections
- Depth visualization and bilinear transformations
- Technical report and assignment writeup

### 🎯 Assignment 2: Feature Detection & Extraction
**Feature Detection & Extraction** - Identifying and extracting meaningful image features
- Edge detection (Canny, Sobel)
- Morphological operations (opening, closing, watershed)
- Object counting and segmentation
- Line detection (Hough transform)
- Image augmentation techniques
- Video processing with feature detection

### 🎨 Assignment 3: Augmented Reality
**Augmented Reality with Planar Homographies** - Applying features for real-world AR
- Feature matching and descriptor extraction (BRIEF, SIFT)
- Planar homography computation
- Real-time video augmentation
- Rotation invariance testing
- Virtual object overlay on planar surfaces

## Technologies
- Python 3
- OpenCV (cv2)
- NumPy
- Matplotlib
- Jupyter Notebooks
- PIL/Pillow

## Repository Structure
Each assignment builds on previous concepts:
1. **Lab 1** establishes camera models and calibration
2. **Assignment 1** explores projection and image formation
3. **Assignment 2** introduces feature detection and image analysis
4. **Assignment 3** combines features with homographies for AR applications

## Getting Started
Each folder contains its own README with specific instructions and details about the implementations.
