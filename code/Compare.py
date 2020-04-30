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

# Global settings
g_modelsFolder = "models/"
g_orig_size = 512
g_small_size = 64
g_num_files = 100 # 100 files per domain
g_upscale_factor = str(g_small_size) + "-" + str(g_orig_size)

# original files are in ../testdata/resized/[domain]/512"
g_origLocation = "../testdata/resized/"
# upscaled files are in "../testdata/upscaled/[domain]/64-512"
g_upscaleLocation = "../testdata/upsampled/"
# output of sr files is "../testdata/sr_[model]/[domain]/64-512"
g_srLocation = "../testdata/sr/"

g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def get_model_list():
    return [f for f in os.listdir(g_modelsFolder) if os.path.isfile(os.path.join(g_modelsFolder, f))]

def get_domain_list():
    return [d for d in os.listdir(g_origLocation) if os.path.isdir(os.path.join(g_origLocation, d))]

def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def inference_img(img, model):
    img = (img.astype(np.float32) / 127.5) - 1
    img = img.transpose((2,0,1))
    t = np.expand_dims(img, 0)
    img_inputs = torch.from_numpy(t).to(g_device)
    prediction = model.test(img_inputs)
    out_img = prediction.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]
    out_img = (np.round(out_img * 127.5 + 127.5)).astype(np.uint8)
    return out_img

def save_img(fname, img):
    ensure_dir_exists(fname)
    img = Image.fromarray(img)
    img.save(fname)

def inference_file(fname, outdir, model):
    img = Image.open(fname)
    
    up_img = inference_img(np.array(img), model)

    out_fname = os.path.join(outdir, os.path.basename(fname))
    save_img(out_fname, up_img)
    return up_img

def compare_imgs(im1, im2, fil_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore skimage's deprecation warinings.
        
        # structural similarity index
        # [ss, im] = metrics.structural_similarity(im1,im2,multichannel=True)
        [ss, im] = metrics.structural_similarity(im1, im2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True, full=True)
        if fil_out:
            im8 = skimage.img_as_ubyte(np.clip(im, -1.0, 1.0))
            save_img(fil_out, im8)

        # mean square error
        mse = metrics.mean_squared_error(im1, im2)

        # normalized root mean squared error
        nrmse = metrics.normalized_root_mse(im1, im2)

        # peak signal-to-noise ratio
        psnr = metrics.peak_signal_noise_ratio(im1, im2)

        return [ss, mse, nrmse, psnr]

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

def write_data(data):
    fname = os.path.join(g_srLocation, "data.csv")
    with open(fname, 'w') as ws:
        ws.write("model,domain,filnum,ssim\n")
        for model_name, domains in data.items():
            for domain_name, fils in domains.items():
                for f in range(len(fils)):
                    ws.write("{},{},{},{}\n".format(model_name, domain_name, f, fils[f]))

def main():
    """main function"""    

    model_names = get_model_list()
    domain_names = get_domain_list()

    # delete existing
    #if os.path.isdir(g_srLocation):
    #    shutil.rmtree(g_srLocation)

    # make a list of all the files to be processed.
    fils = dict()
    for domain_name in domain_names:
        fils[domain_name] = [os.path.basename(f) for f in glob.glob(os.path.join(g_origLocation, domain_name, str(g_orig_size), "*.jpg"))]

    data = read_data()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore skimage's deprecation warinings.

        # compute upsampled image metrics
        model_name = 'upsampling'
        if not model_name in data:
            data[model_name] = dict()
            for domain_name in domain_names:
                data[model_name][domain_name] = np.zeros(len(fils[domain_name]))
                for f in range(len(fils[domain_name])):
                    file_name = fils[domain_name][f]
                    fname_orig = os.path.join(g_origLocation, domain_name, str(g_orig_size), file_name)
                    fname_upscale = os.path.join(g_upscaleLocation, domain_name, g_upscale_factor, file_name)
                    img_orig = np.array(Image.open(fname_orig))
                    img_up = np.array(Image.open(fname_upscale))
                    metrics = compare_imgs(img_orig, img_up, None)
                    data[model_name][domain_name][f] = metrics[0]


        for m in range(len(model_names)):
            model_name = model_names[m]
            if not model_name in data:
                data[model_name] = dict()

            model = SRNet(image_shape=(64,64,3),device=g_device,continue_from_save=False)
            model.load(os.path.join(g_modelsFolder, model_name))
            for d in range(len(domain_names)):
                domain_name = domain_names[d]
                if domain_name in data[model_name]:
                    continue

                print("Infering and comparing model {} on domain {}...".format(model_name, domain_name))
                data[model_name][domain_name] = np.zeros(len(fils[domain_name]))
                sr_path = os.path.join(g_srLocation, model_name, domain_name, g_upscale_factor)
                for f in range(len(fils[domain_name])):
                    file_name = fils[domain_name][f]
                    fname_orig = os.path.join(g_origLocation, domain_name, str(g_orig_size), file_name)
                    fname_small = os.path.join(g_origLocation, domain_name, str(g_small_size), file_name)
                    
                    img_orig = np.array(Image.open(fname_orig))
                    img_sr = inference_file(fname_small, sr_path, model)
                    metrics = compare_imgs(img_orig, img_sr, None)
                    data[model_name][domain_name][f] = metrics[0]

    write_data(data)

if __name__ == '__main__':
    main()
