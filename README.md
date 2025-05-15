# Simple MTG Feature Extraction

## Overview

This is a MTG feature extration workflow targeting a production ready scope for fast deployment and a small footprint.

## Scope
- Generate candidate_features.h5, faiss_ivf.index, and id_map.json
- Do not require images to be downloaded to disk as a part of the feature extraction

## Rough Workflow Outline

1. **Download Card Image to memory**
2. **Extract Features w/ SIFT and Store in HDF5 as a batch**
3. **Build FAISS Index**
4. **Inference and Evaluate**

## Citations

This project is heavily built upon the research detailed in the thesis paper [*Magic: The Gathering Card Reader with Feature Detection* by Dakota Madden-Fong (2018)](https://github.com/TrifectaIII/MTG-Card-Reader/blob/master/Thesis%20Paper%20-%20MTG%20Card%20Reader.pdf), which significantly inspired the design and basis of this project.
