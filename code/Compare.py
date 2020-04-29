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
from PIL import Image
from models import *
import shutil

# Global settings
g_modelsFolder = "models/"
g_orig_size = 512
g_small_size = 64
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

#def compare_using_model(model, files):

def main():
    """main function"""    

    model_names = get_model_list()
    domain_names = get_domain_list()

    # delete existing
    if os.path.isdir(g_srLocation):
        shutil.rmtree(g_srLocation)

    # make a list of all the files to be processed.
    fils = dict()
    for domain_name in domain_names:
        fils[domain_name] = [os.path.basename(f) for f in glob.glob(os.path.join(g_origLocation, domain_name, str(g_orig_size), "*.jpg"))]

    stats = np.zeros([len(model_names), len(domain_names), len(fils[domain_names[0]]), 2]) # assume all domains have same num of files
    means = np.zeros([len(model_names), len(domain_names), 2]) # assume all domains have same num of files

    for m in range(len(model_names)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore skimage's deprecation warinings.
            model_name = model_names[m]
            model = SRNet(image_shape=(64,64,3),device=g_device,continue_from_save=False)
            model.load(os.path.join(g_modelsFolder, model_name))
            for d in range(len(domain_names)):
                domain_name = domain_names[d]
                sr_path = os.path.join(g_srLocation, model_name, domain_name, g_upscale_factor)
                for f in range(len(fils[domain_name])):
                    file_name = fils[domain_name][f]
                    fname_orig = os.path.join(g_origLocation, domain_name, str(g_orig_size), file_name)
                    fname_upscale = os.path.join(g_upscaleLocation, domain_name, g_upscale_factor, file_name)
                    fname_small = os.path.join(g_origLocation, domain_name, str(g_small_size), file_name)
                    
                    img_orig = np.array(Image.open(fname_orig))
                    img_up = np.array(Image.open(fname_upscale))
                    img_sr = inference_file(fname_small, sr_path, model)
                    metrics_base = compare_imgs(img_orig, img_up, None)
                    metrics_sr = compare_imgs(img_orig, img_sr, None)
                    stats[m, d, f, 0] = metrics_base[0]
                    stats[m, d, f, 1] = metrics_sr[0]
                means[m, d, 0] = np.mean(stats[m, d, :, 0])
                means[m, d, 1] = np.mean(stats[m, d, :, 1])
                print("Domain,{},model,{},{},{}".format(domain_name, model_name, means[m,d,0], means[m,d,1]))

    print(stats)
    np.save(os.path.join(g_srLocation), "stats.nparr", stats)
    np.save(os.path.join(g_srLocation), "means.nparr", stats)

    

    # stats = np.empty([len(fils), 8])
    # cnt = 0
    # for fil in fils:
    #     # find the corresponding file in source2Location
    #     try:
    #         fil_out = os.path.join(f_resultsLocation, os.path.basename(fil[0]))
    #         im_src = cv2.imread(fil[0])
    #         im_base = cv2.imread(fil[1])
    #         im_sr = cv2.imread(fil[2])
    #         metrics_base = compare_imgs(im_src, im_base, None)
    #         metrics_sr = compare_imgs(im_src, im_sr, None)
    #         stats[cnt,:] = metrics_base + metrics_sr
    #         # print("{}: {}".format(os.path.basename(fil[0]), metrics))
    #     except Exception as ex:
    #         # store the errors in a list, to make it easier to see how things finished
    #         errors.append((fil[0], ex))
    #         print(f" error: {ex}")
    #     cnt = cnt + 1

    # plot_results(stats)

    # # print out the list of errors at the end
    # cnt = 1
    # print(f"{len(errors)} errors occurred.")
    # for err in errors:
    #     print(f"Error {cnt}: ({err[0]}) {err[1]}")
    #     cnt = cnt + 1

    # means = np.mean(stats, axis=0)
    # stddevs = np.std(stats, axis=0)
    # print(means)
    # print(stddevs)

    # np.savetxt(os.path.join(f_resultsLocation, 'stats.csv'), stats, delimiter=',')

if __name__ == '__main__':
    main()
