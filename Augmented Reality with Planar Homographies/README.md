# Assignment 3: Augmented Reality with Planar Homographies

This assignment applies feature detection and matching to create real-time augmented reality applications using planar homography transformations.

## What's Inside

### Main Implementation
- **mkoome/code/** - Complete AR implementation with:
  - **ar.py** - Main augmented reality application
  - **planarH.py** - Planar homography computation and transformation
  - **matchPics.py** - Feature matching between reference and query images
  - **HarryPotterize.py** - Specific AR application (virtual object overlay)
  - **briefRotTest.py** - Testing BRIEF descriptor rotation invariance
  - **classify.py** - Image classification utilities
  - **helper.py** - Helper functions for AR pipeline
  - **loadVid.py** - Video loading and processing utilities

### Key Techniques
- **Feature Matching**: BRIEF and SIFT descriptors for finding correspondences
- **Planar Homography**: Computing 3x3 transformation matrix between planar surfaces
- **Rotation Invariance**: Testing feature robustness under rotation
- **Real-time Video Processing**: Frame-by-frame augmentation
- **Virtual Object Overlay**: Placing 3D objects on detected planar surfaces

### Data & Results
- **mkoome/data/** - Reference images and video sequences for AR
- **mkoome/results/** - AR output results and augmented frames
- **HW3 Assets/** - Assignment resources and reference materials

### Generated Outputs
- **Feature Matching Results**: Feature Matching.png, Feature Matching_2.png
- **Rotation Tests**: 10_deg.png, 90_deg.png, 180_deg.png, 350_deg.png
- **Augmentation Results**: augmentation.png, Final_composite.png, ar_snap.png
- **Analysis**: Histogram.png, faces.png
- **Descriptor Tests**: 0_1_sigma_0_55_ratio.png, 0_1_sigma_0_65ratio.png

### Documentation
- **ACV_Spring_2026_Ass3_Writeup.pdf** - Assignment specifications
- **mkoome.pdf** - Detailed report on implementation and results
- **requirements.txt** - Python dependencies

## Purpose
Combine feature detection, matching, and geometric transformations to build practical augmented reality systems. Learn how to detect planar surfaces in real-time video and overlay virtual content accurately.

## Key Concepts
- **Homography**: Geometric transformation between two planes
- **Feature Correspondence**: Matching features across images
- **Descriptor Robustness**: Testing invariance to rotation and scale
- **Real-time AR**: Processing video frames for live augmentation
- **Geometric Verification**: Using RANSAC to filter outliers

## Workflow
1. Detect features in reference image
2. Match features between reference and query frames
3. Compute planar homography from matched points
4. Apply homography to overlay virtual content
5. Handle rotation and scale variations
