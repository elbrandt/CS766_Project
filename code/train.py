import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time

from models import *
from loaders import *

num_epochs      = 1000
batch_size      = 32
num_samples     = 24000
lr              = 1e-3
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_shape     = (4,480,640)
print_interval  = 1    #in terms of batches
save_interval   = 1    #in terms of epochs
img_progress_interval   = 50    #in terms of batches

continue_from_save = False

#default datatype
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    print("Starting training...")

    # domain_A_dir = "/media/amelmquist/easystore/datasets/sidd/256_noise_train/"
    # domain_B_dir = "/media/amelmquist/easystore/datasets/sidd/256_noise_train/"

    # domain_A_dir = "/srv/home/amelmquist/datasets/hall/synthetic/train/"
    # domain_B_dir = "/srv/home/amelmquist/datasets/hall/real/train/"

    small_dir = "/home/amelmquist/datasets/sr/Buildings/64/"
    large_dir = "/home/amelmquist/datasets/sr/Buildings/128/"

    data_JIT_loader = Img2ImgJITLoader(small_dir,large_dir,paired_samples=True,num_samples=num_samples,file_type=".jpg")
    data_loader = torch.utils.data.DataLoader(data_JIT_loader, batch_size=batch_size,shuffle=True,num_workers=4)

    #intialize the super resolution network
    model = SRNet(image_shape=image_shape,device=device,continue_from_save=continue_from_save)

    model.train(data_loader,num_epochs=num_epochs,batch_size=batch_size,num_samples=num_samples,lr=lr,print_interval=print_interval,save_interval=save_interval,img_progress_interval=img_progress_interval)

    model.save()
