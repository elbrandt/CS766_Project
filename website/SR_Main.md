# Super Resolution Image Enhancement
## UW Madison CS766 - Computer Vision, Spring 2020

Asher Elmquist (```amelmquist@wisc.edu```)

Eric Brandt (```elbrandt@wisc.edu```)

## Overview

The purpose of this project is to explore and better understand image super-resolution. We begin by preparing a large, high resolution, labeled training data set and then implementing and training a convolutional neural network for super-resolution based on the current state of the art literature. Our investigation set out to answer three questions:
1. Can we train a state-of-the-art CNN to produce Super Resolution images and how can we assess the performance?
2. Does the the image domain (as identified by primary image label) of the training data make a difference in the training results? What about in the test images being inferenced for Super Resolution?
3. Does applying Super Resolution to low resolution images improve the accuracy of downstream processes such as image segmentation?

## Table of Contents
- [Super Resolution Image Enhancement](#super-resolution-image-enhancement)
  - [Overview](#overview)
  - [Introduction](SR_Introduction.md)
  - [Building the SR Network](SR_Building.md)
    - [Training Dataset](SR_Building.md#training-dataset)
    - [Network Structure](SR_Building.md#network-structure)
  - [Inferencing Results](SR_Results.md)
    - [Subjective Super-Resolution Results](SR_Results.md#subjective-super-resolution-results)
    - [Objective Performance Measure - SSIM](SR_Results.md#objective-performance-measure---ssim)
  - [Domain Specific Training and Inferencing](SR_DomainSpecific.md)
  - [SR and Image Segmentation](SR_Segmentation.md)
  - [Conclusion](SR_Conclusion.md)
    - [Future Work](SR_Conclusion.md#future-work)
    - [References](SR_Conclusion.md#references)

---

|Prev: [Main](SR_Main.md) | Up: [Main](SR_Main.md) | Next: [Introduction](SR_Introduction.md) |
