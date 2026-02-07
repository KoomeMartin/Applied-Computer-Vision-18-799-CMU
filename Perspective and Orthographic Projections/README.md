# Perspective and Orthographic Projections

This folder contains an assignment implementation exploring 3D-to-2D projection techniques and image formation in computer vision.

## What's Inside

### Main Implementation
- **mkoome.ipynb** - Jupyter notebook implementing perspective and orthographic projection algorithms, including:
  - Standard plane projection (focal length-based)
  - Tilted plane projection (arbitrary plane equations)
  - Bilinear transformations
  - Depth visualization and color mapping
  - Camera orbit animations

### Data Files
- **points.pkl** / **data.pkl** - 3D point cloud data for projection experiments
- **scene.ply** / **mkoome_mesh.ply** - 3D mesh files for visualization

### Generated Outputs
- **orbit_frames/** - 120 frames of camera orbiting around a 3D scene
- **views/** - Orthographic projections from 6 standard viewpoints (front, back, left, right, top, bottom) plus paired comparison images
- **paired_views/** - Empty folder for additional view comparisons
- Various PNG files showing projection results (bilinear transformations, depth maps, perspective vs orthographic comparisons)

### Documentation
- **ACV_Spring_2026_Ass1_Writeup.pdf** - Assignment specifications
- **Technical Report.pdf** - Detailed report on implementation and results
- **Camera Orbit Video - Made with Clipchamp.mp4** - Compiled animation of orbit frames

## Purpose
Demonstrates fundamental concepts of 3D projection, camera models, and image formation for computer vision applications.
