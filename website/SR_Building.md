# Super Resolution Image Enhancement
###### UW Madison CS766 - Computer Vision, Spring 2020

| Asher Elmquist (```amelmquist@wisc.edu```) | Eric Brandt (```elbrandt@wisc.edu```) |

|Prev: [Introduction](SR_Introduction.md) | Up: [Main](SR_Main.md) | Next: [Inferencing Results](SR_Results.md) |

# Building the Super Resolution Network
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.



## Training Dataset
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Network Structure
The network used for super-resolution was based on ResNet and upsampled using
a pixel shuffle operation which is fully learnable. 4 ResNet blocks were used
with instance normalization and relu activation. The output from the ResNet
block was summed with the input per typical ResNet use. 3 successive upscaling
operations were used, each with factor of 2 for total image upscaling factor
of 8.


<p align="center">
  <img src="images/network/network_structure.png" width="200">
</p>

The network was trained using a combination of pixel loss and feature loss.
Pixel loss is defined as the mean squared error between the images. Feature loss
was based on the 4th layer of the pre-trained VGG19 network (relu_1_2), and
computed as the MSE between the features in the prediction and the features in
the target. While pixel-error minimizes image error, feature loss had been shown
to promote high-frequency reconstruction.

The full loss function is given by:

<img src="images/network/eqn_loss.png" width="500">

where x and y are the prediction and target images.


|Prev: [Introduction](SR_Introduction.md) | Up: [Main](SR_Main.md) | Next: [Inferencing Results](SR_Results.md) |
