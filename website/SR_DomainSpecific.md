# Super Resolution Image Enhancement
###### UW Madison CS766 - Computer Vision, Spring 2020

|Prev: [Inferencing Results](SR_Results.md) | Up: [Main](SR_Main.md) | Next: [SR and Image Segmentation](SR_Segmentation.md) |

# Domain Specific Training and Inferencing
Having established that we can achieve good results with a Super Resolution network, we investigated two questions about the types of images used for Super Resolution. 
1. What are the effects of training the the CNN model purely on images having a specific label. For example, how would a CNN model trained purely on 'Building' images compare to a CNN model trained purely on 'Food' images.
2. Given a trained model, does its performance differ based on the types of images it is tested against? For example, does a CNN model trained on 'Building' images perform better on test 'Building' images than it does on test 'Flower' images? 

To answer these questions, we trained four separate CNN models. Each model was trained exclusively with 10,000 images containing a single primary label.  We evaluated the performance of each model by using the 'Normalized SSIM' described [here](SR_Results.md#objective-performance-measure---ssim).

During training, the model was evaluated periodically to determine the (rough) optimal amount of training time, watching for indications of overtraining. As an example, we can see that the when training the model against 'Dog' labeled images, we empirically observed the best results were achieved around 248 presentations of the 10,000 images (epochs). In this chart, we see the performance relative to linear interpolation ('upsampling') of the 'Dog' model on all test data sets, when the training was stopped after 140, 162, 248, 302, and 409 epochs. Each bar in the chart represents the mean of the Normalized SSIM of 100 training images of the specified label.

<p align="center">
  <img src="images/results/dog_training_epochs.png">
</p>

After training and monitoring the progress of all four CNN models, we compared the four models by testing them on the four separate test data sets. The results are shown in the following chart:

<p align="center">
  <img src="images/results/domain_transfer.png">
</p>




---

|Prev: [Inferencing Results](SR_Results.md) | Up: [Main](SR_Main.md) | Next: [SR and Image Segmentation](SR_Segmentation.md) |

Asher Elmquist (```amelmquist@wisc.edu```), Eric Brandt (```elbrandt@wisc.edu```) 2020
