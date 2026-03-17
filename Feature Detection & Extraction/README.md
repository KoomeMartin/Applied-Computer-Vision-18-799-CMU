# Assignment 2: Feature Detection & Extraction

This assignment explores fundamental image processing techniques for detecting and extracting meaningful features from images and videos.

## What's Inside

### Main Implementation
- **mkoome_Assignment2.ipynb** - Complete assignment notebook covering:
  - **Q1: Object Counting** - Using morphological operations and watershed segmentation to count objects in images
  - **Q2: Edge Detection** - Comparing Canny and Sobel edge detection methods
  - **Q3: Line Detection** - Hough transform for detecting lines in images

### Key Techniques Implemented
- **Edge Detection**: Canny edge detector, Sobel operator
- **Morphological Operations**: Opening, closing, erosion, dilation
- **Segmentation**: Watershed algorithm for object separation
- **Line Detection**: Hough transform for line identification
- **Image Augmentation**: Rotation, flipping, cropping, brightness, blur, noise

### Data & Results
- **q1_data/** - Sample images for object counting task
- **q2_data/** - Images for edge detection comparison
- **q3_data/** - Images for line detection with Hough transform
- **augmented_images/** - Results of various image augmentation techniques
- **mkoome/** - Complete working directory with all outputs and intermediate results

### Generated Outputs
- **Edge Detection Results**: canny.png, sobel_canny.png, canny_houge.png
- **Morphological Operations**: binary.png, opening.png, closing.png
- **Segmentation**: watershed_seg.png, sure_fg.png, dist.png
- **Video Processing**: processed_video_canny_1.mp4, processed_video_watershed.mp4
- **Augmentation Examples**: Various transformed versions of original images

### Documentation
- **ACV Spring 2026 Ass2 Writeup-1.pdf** - Assignment specifications and requirements

## Purpose
Learn to detect and extract visual features that are fundamental to higher-level computer vision tasks like object recognition, tracking, and augmented reality. These techniques bridge the gap between raw pixel data and meaningful image understanding.

## Key Concepts
- **Feature Detection**: Identifying important points/edges in images
- **Feature Extraction**: Computing descriptors for detected features
- **Image Processing Pipeline**: Preprocessing → Detection → Analysis
- **Robustness**: Testing feature detection under various conditions
