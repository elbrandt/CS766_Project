import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from models import *
from loaders import *

batch_size      = 1
num_samples     = 10
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

    dir_64 = "/home/amelmquist/datasets/sr/Buildings/test/64/"
    # dir_128 = "/home/amelmquist/datasets/sr/Buildings/128/"
    # dir_256 = "/home/amelmquist/datasets/sr/Buildings/256/"
    dir_512 = "/home/amelmquist/datasets/sr/Buildings/test/512/"
    dir_512_up = "/home/amelmquist/datasets/sr/upsampled/Buildings/64-512/"


    data_JIT_loader = Img2ImgJITTestLoader(dir_64,dir_512,dir_512_up,paired_samples=True,num_samples=num_samples,file_type=".jpg")
    data_loader = torch.utils.data.DataLoader(data_JIT_loader, batch_size=batch_size,shuffle=True,num_workers=4)

    #intialize the GAN
    model = SRNet(image_shape=image_shape,device=device,continue_from_save=False)
    model.load()

    for i,data in enumerate(data_loader, 0):
        low_res,high_res,upsampled = data
        img_inputs = low_res.to(device)
        # for u in range(1,4):
        #     u_f = 2**u
        #     img_inputs.append(torch.from_numpy(np.random.uniform(-1,1,(low_res.shape[0],1,low_res.shape[3]*u_f,low_res.shape[3]*u_f)).astype(np.float32)).to(device))

        # low_res = low_res.to(device)
        start = time.time()
        prediction = model.test(img_inputs)
        end = time.time()
        print("Inference in:",end-start,"seconds")

        f, ax = plt.subplots(1,4,figsize=(12,4))
        ax[0].imshow(low_res.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[0].set_title('Low Resolution')

        ax[1].imshow(prediction.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[1].set_title('SR Prediction')
        # print("max:",np.max(prediction.detach().cpu().numpy()))
        # print("min:",np.min(prediction.detach().cpu().numpy()))

        ax[2].imshow(upsampled.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[2].set_title('OpenCV Upsampled')

        ax[3].imshow(high_res.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
        ax[3].set_title('High Resolution')


        plt.show()
        # plt.savefig("results/test_img_"+str(i)+".png")
        plt.close()

        # if(i>10):
        #     break



    #
