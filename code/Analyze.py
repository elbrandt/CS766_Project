#
# Compare Image Files from 2 directory trees
# Eric Brandt, Asher Elmquist
#
#
import os
import cv2
import glob
import skimage
import torch
import torch.nn as nn
import torch.optim as optim
import skimage.metrics as metrics
import numpy as np
import warnings
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from models import *

g_srLocation = "../testdata/sr/"
g_num_files = 100

def plot_results(stats):
    fig = plt.figure()
    axs = fig.subplots(2, 2)
    titles = ["Structural Similarity Index", "Mean Square Error", "Normalized Mean Square Error", "Peak SNR"]
    n = 0
    for r in range(2): 
        for c in range(2):
            ax = axs[r,c]
            ax.plot(stats[:,n], label="Linear interp. upscale")
            ax.plot(stats[:,n+4], label="Super Resolution")
            ax.set_title(titles[n])
            n = n + 1
    fig.suptitle(f_domain)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    plt.show()

def read_data():
    data = dict()
    fname = os.path.join(g_srLocation, "data.csv")
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            lin_num = 0
            for lin in f:
                lin_num = lin_num + 1
                if lin_num == 1:  # skip header
                    continue
                lst = lin.split(',')
                model = lst[0]
                domain = lst[1]
                filnum = int(lst[2])
                ssim = float(lst[3])
                if not model in data: # model
                    data[model] = dict()
                if not domain in data[model]:  # domain
                    data[model][domain] = np.zeros(g_num_files)
                data[model][domain][filnum] = ssim
    return data

def chart_mean(model_names, domain_names, means):
    fig,ax = plt.subplots()
    nmodels = len(model_names)
    dx = 0.8 / nmodels
    offsets = np.linspace(-0.4 + dx/2, 0.4 - dx/2, nmodels) 
    for d in range(len(domain_names)):
        for m in range(len(model_names)):
            print("[{}-{}]\n".format(d + offsets[m], d+offsets[m]+dx))
            ax.bar(d + offsets[m], means[m,d], width=dx, color="C{}".format(m))
    #ax.legend(labels=model_names, loc='lower right')
    ax.legend(labels=model_names, bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("Image Domain")
    ax.set_ylabel("Structural Similarity, Normalized to 'upsampling'=1.0")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(domain_names)
    ax.set_title("Structural Similarity Index by Model and Domain")
    ax.set_ylim([0.9, 1.10])
    plt.tight_layout()
    plt.show()

def main():
    """main function"""    

    data = read_data()

    #model_names = list(data.keys())
    #model_names = ["upsampling", "Food_200", "Food_239", "Food_334", "Food_386", "Food_596", "Food_726"]
    #model_names = ["upsampling", "Flower_140", "Flower_165", "Flower_253", "Flower_308"]
    #model_names = ["upsampling", "Dog_140", "Dog_162", "Dog_248", "Dog_302"]
    #model_names = ["upsampling", "Building_200", "Building_317", "Building_380", "Building_631", "Building_787"]
    model_names = ["upsampling", "Building_380", "Dog_248", "Flower_308", "Food_726"]
    num_models = len(model_names)
    domain_names = ["Food", "Dog", "Building", "Flower"]
    num_domains = len(domain_names)

    # calculate means
    means = np.zeros([num_models, num_domains])
    m_upsampling = -1
    for m in range(num_models):
        for d in range(num_domains):
            means[m, d] = np.mean(data[model_names[m]][domain_names[d]])
        if model_names[m] == 'upsampling':
            m_upsampling = m
    
    # normalize
    for d in range(num_domains):
        divisor = means[m_upsampling, d]
        for m in range(num_models):
            means[m, d] = means[m, d] / divisor


    chart_mean(model_names, domain_names, means)
        


if __name__ == '__main__':
    main()