# import torchvision
import numpy as np
import torch
import torch.utils.data
import glob
from PIL import Image
import matplotlib.pyplot as plt
import random

class Img2ImgJITLoader(torch.utils.data.Dataset):

    def __init__(self,x_dir,y_dir,file_type=".png",paired_samples=True,num_samples=-1):
        self.paired_samples = paired_samples

        self.x_dir = x_dir
        self.y_dir = y_dir

        #only know which images can be loaded, don't actually load them yet
        self.img_list_x = glob.glob(x_dir+"*"+file_type)
        self.img_list_y = glob.glob(y_dir+"*"+file_type)

        self.data_len = min(len(self.img_list_x),len(self.img_list_y))

        if(num_samples > -1):
            self.data_len = num_samples
            self.img_list_x = self.img_list_x[0:num_samples]
            self.img_list_y = self.img_list_y[0:num_samples]

        print("Number of samples in dataset: x=",len(self.img_list_x), "y=", len(self.img_list_y))

    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        #shuffle both lists independently if we the samples shouldn't be paired
        if not self.paired_samples:
            random.shuffle(self.img_list_x)
            random.shuffle(self.img_list_y)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #now load the files requested
        data_x = (np.array(Image.open(self.img_list_x[idx])).astype(np.float32) / 127.5) - 1
        data_y = (np.array(Image.open(self.img_list_y[idx])).astype(np.float32) / 127.5) - 1

        data_x = data_x.transpose((2,0,1))
        data_y = data_y.transpose((2,0,1))

        images_x = torch.from_numpy(data_x)
        images_y = torch.from_numpy(data_y)

        return images_x, images_y