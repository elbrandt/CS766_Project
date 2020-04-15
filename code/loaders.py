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
        self.img_list_y = []
        for f in self.img_list_x:
            file_name = f.split("/")[len(f.split("/"))-1]
            self.img_list_y.append(y_dir+file_name)

        # self.img_list_y = glob.glob(y_dir+"*"+file_type)

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

class Img2ImgJITTestLoader(torch.utils.data.Dataset):

    def __init__(self,x_dir,y_dir,up_dir,file_type=".png",paired_samples=True,num_samples=-1):
        self.paired_samples = paired_samples

        self.x_dir = x_dir
        self.y_dir = y_dir
        self.up_dir = up_dir

        #only know which images can be loaded, don't actually load them yet
        self.img_list_x = glob.glob(x_dir+"*"+file_type)
        self.img_list_y = glob.glob(y_dir+"*"+file_type)
        self.img_list_up = glob.glob(up_dir+"*"+file_type)

        self.data_len = min(len(self.img_list_x),len(self.img_list_y),len(self.img_list_up))

        if(num_samples > -1):
            self.data_len = num_samples
            self.img_list_x = self.img_list_x[0:num_samples]
            self.img_list_y = self.img_list_y[0:num_samples]
            self.img_list_up = self.img_list_up[0:num_samples]

        print("Number of samples in dataset: x=",len(self.img_list_x), "y=", len(self.img_list_y),"up=", len(self.img_list_up))

    def __len__(self):
        return self.data_len

    def __getitem__(self,idx):
        #shuffle both lists independently if we the samples shouldn't be paired
        if not self.paired_samples:
            random.shuffle(self.img_list_x)
            random.shuffle(self.img_list_y)
            random.shuffle(self.img_list_up)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #now load the files requested
        data_x = (np.array(Image.open(self.img_list_x[idx])).astype(np.float32) / 127.5) - 1
        data_y = (np.array(Image.open(self.img_list_y[idx])).astype(np.float32) / 127.5) - 1
        data_up = (np.array(Image.open(self.img_list_up[idx])).astype(np.float32) / 127.5) - 1

        data_x = data_x.transpose((2,0,1))
        data_y = data_y.transpose((2,0,1))
        data_up = data_up.transpose((2,0,1))

        images_x = torch.from_numpy(data_x)
        images_y = torch.from_numpy(data_y)
        images_up = torch.from_numpy(data_up)

        return images_x, images_y,images_up
