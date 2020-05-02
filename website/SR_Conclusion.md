# Super Resolution Image Enhancement
###### UW Madison CS766 - Computer Vision, Spring 2020

|Prev: [SR and Image Segmentation](SR_Segmentation.md) | Up: [Main](SR_Main.md) | Next: [Conclusion](SR_Conclusion.md) |

# Conclusion
This project was conducted in order to begin studying machine learned super-resolution, its sensitivity to domain specific image scaling, and the application of SR in improving sub-pixel segmentation. While only a component of the project goal, presented here is an implementation of super-resolution that leverages state-of-the-art techniques in CNN based image augmentation including learned upsampling and feature loss. No claim is made here that this technique represents the state-of-the-art implementation or results. The implementation was sufficient enough to perform downstream studies with the implementation consistently outperforming traditional upsampling methods in structural similarity to the ground-truth ([Objective Results](SR_Results.md#objective-performance-measure---ssim)). For the domain specific study, it was found that the domain can influence the training and inference performance depending on the images ([Domain Results](SR_DomainSpecific.md)). Lastly, for performance in improving segmentation of upsampled images (equivalent to sub-pixel segmentation), it was found that the SR network, even with improved SSIM does not result in higher accuracy segmentation. In fact, the segmentation results using super-resolved images was signifacntly more error prone ([Segmentation Results](SR_Segmentation.md)). As a result, further work needs to be done for understanding super resolution as an assitive measure for downstream tasks such as segmentation and object detection as improved SSIM and subjective quality does not necessarily correlate to improved accuracy for segmentation.

## Future Work
Further work on improving the super-resolution network could be done. A few additional approaches such as pre-upscaling, progressive upsampling, and edge-aware methods could be used. Additionally, a generative adversarial network could be used in training the SR network to further promote high-frequency content. A discriminator that uses the first layers from a pretrained ResNet could improve SR for segmentation as the loss function would share the same feature maps as the segmentation metric. For a deeper dive into machine learning theory, hyperparameter studies could be performed. Another direction of future work could be to apply object detection to super-resolved images. This is an extension of applying segmentation but would allow a study of if SR networks can improve detectability or improve the precision of pose estimation.


## References

All references and citations in this project and website are live links embedded in-line in pages of this site. A formal, formatted bibliography can be found at the end of the mid-term project report [here](https://github.com/elbrandt/CS766_Project/blob/master/mid-term/CS766_ElmquistBrandt_Super_Resolution_Image_Enhancement_MidTerm.pdf).

---

|Prev: [SR and Image Segmentation](SR_Segmentation.md) | Up: [Main](SR_Main.md) | Next: [Conclusion](SR_Conclusion.md) |

Asher Elmquist (```amelmquist@wisc.edu```), Eric Brandt (```elbrandt@wisc.edu```) 2020
