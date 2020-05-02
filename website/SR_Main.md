# Super Resolution Image Enhancement
## UW Madison CS766 - Computer Vision, Spring 2020

Asher Elmquist (```amelmquist@wisc.edu```)

Eric Brandt (```elbrandt@wisc.edu```)

## Overview

The purpose of this project is to explore and better understand image super-resolution. We begin by preparing a large, high resolution, labeled training data set annd then implementing and training a convolutional neural network for super-resolution based on the current state of the art literature. After assessing the subjective quality of the inferencing of the network we created and trained, we probe its characteristics with two specific investigations. First, we explore the 'domain sensitivity' to both the training and the inferencing of the network using objective performance measures. Second, we explore the utility of super-resolution inferencing to downstream image processing tasks, using the specific example of image segmentation.

## Table of Contents
- [Super Resolution Image Enhancement](#super-resolution-image-enhancement)
  - [UW Madison CS766 - Computer Vision, Spring 2020](#uw-madison-cs766---computer-vision-spring-2020)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
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
