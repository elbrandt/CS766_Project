import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time

from models import *
from loaders import *

batch_size      = 1
num_samples     = -1
# lr              = 1e-4
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_shape     = (4,480,640)
# print_interval  = 10    #in terms of batches
# save_interval   = 1    #in terms of epochs

# continue_from_save = True

#default datatype
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    print("Starting training...")

    # domain_A_dir = "/media/amelmquist/easystore/datasets/sidd/256_noise_test/"
    # domain_B_dir = "/media/amelmquist/easystore/datasets/sidd/256_noise_test/"

    domain_A_dir = "/home/amelmquist/datasets/hall/synthetic/test/"
    domain_B_dir = "/home/amelmquist/datasets/hall/real/test/"

    data_JIT_loader = Img2ImgJITLoader(domain_A_dir,domain_B_dir,paired_samples=False,num_samples=num_samples)
    data_loader = torch.utils.data.DataLoader(data_JIT_loader, batch_size=batch_size,shuffle=True,num_workers=4)

    #intialize the GAN
    model = CyclePatchGAN(image_shape=image_shape,device=device,continue_from_save=False)
    model.load()

    for i,data in enumerate(data_loader, 0):
        real_A,real_B = data
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        fake_a, fake_b, pred_real_a, pred_fake_a, pred_real_b, pred_fake_b = model.test(real_A, real_B)

        print("D(real_a)=%.3f | D(fake_a)=%.3f | D(real_b)=%.3f | D(fake_b)=%.3f" % (pred_real_a, pred_fake_a, pred_real_b, pred_fake_b))

        f, ax = plt.subplots(2,2)
        ax[0,0].imshow(real_A.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[0,0].set_title('Real A')

        ax[0,1].imshow(fake_a.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[0,1].set_title('Fake A')

        ax[1,0].imshow(fake_b.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[1,0].set_title('Fake B')

        ax[1,1].imshow(real_B.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[1,1].set_title('Real B')

        plt.show()




    #
