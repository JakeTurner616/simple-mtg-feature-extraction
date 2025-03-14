# Simple MTG Feature Extraction

## Overview

This is my attempt at a simplified feature extraction workflow for fast and accurate card identification at inference time. With this workflow, I set out to accomplish a few simple tasks:

## Scope
- Batch process all heavy operations – don't fill RAM.
- Use optimized file formats and compression techniques for storing keypoints in high-dimensional space.
- Use quantization and geometric verification for rejecting keypoint outliers.

## Rough Workflow Outline

1. **Download Card Images**
2. **Extract Features w/ SIFT and Store in HDF5**
3. **Build FAISS Index**
4. **Inference and Evaluate**

# TODO

- ~~GUI implementation for real-time identification using a simple YOLO segmentation model for ROI.~~
- Download the 2.5~ GB of resources from a cloud source at runtime and use them. Create a release without Github LFS.
## Citations

This project is heavily built upon the research detailed in the thesis paper [*Magic: The Gathering Card Reader with Feature Detection* by Dakota Madden-Fong (2018)](https://github.com/TrifectaIII/MTG-Card-Reader/blob/master/Thesis%20Paper%20-%20MTG%20Card%20Reader.pdf), which significantly inspired the design and basis of this project.
