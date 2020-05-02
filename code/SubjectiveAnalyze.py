#
# Compare Image Files from 2 directory trees
# Eric Brandt, Asher Elmquist
#
#
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import skimage.io
import skimage.util
from skimage import data

# original files are in ../testdata/resized/[domain]/512"
g_origLocation = "../testdata/resized/"
g_upscaleLocation = "../testdata/upsampled/"
g_srLocation = "../testdata/sr/sr_all_72"

g_images = [
    ( "Building", "000d6781b9abfae8.jpg" ),
    ( "Dog", "8a19c3b7d2007355.jpg" ),
    ( "Flower", "2d27fecf7183a2d7.jpg" ),
    ( "Food", "2d4a38d39b4db37d.jpg" )
]

def compare4x4():
    fig = plt.figure(figsize=[8, 8], dpi=150)
    axs = fig.subplots(4, 4)
    c_titles = ["Low Res", "Lin. Interp", "SR", "Original Hi Res"]
    for r in range(4): 
        f_low = os.path.join(g_origLocation, g_images[r][0], "64", g_images[r][1])
        f_up = os.path.join(g_upscaleLocation, g_images[r][0], "64-512", g_images[r][1])
        f_sr = os.path.join(g_srLocation, g_images[r][0], "64-512", g_images[r][1])
        f_hi = os.path.join(g_origLocation, g_images[r][0], "512", g_images[r][1])
        imgs = [f_low, f_up, f_sr, f_hi]
        for c in range(4):
            ax = axs[r,c]
            img = Image.open(imgs[c])
            ax.imshow(img)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            if r == 3:
                ax.set_xlabel(c_titles[c])
    fig.subplots_adjust(left=0.03, right=1.0, bottom=0.03, top=0.95, wspace=0.0, hspace=0.09)
    fig.suptitle("Subjective Comparison - Linear Interpolation vs SR")
    plt.savefig("../website/images/results/subjective_compare_4x4.png", bbox_inches='tight')
    plt.show()

def compare1x2():
    fig = plt.figure(figsize=[8, 4], dpi=150)
    axs = fig.subplots(1, 2)
    c_titles = ["Linear Interpolation Upscaling", "Super Resolution Result"]
    f_up = os.path.join(g_upscaleLocation, g_images[2][0], "64-512", g_images[2][1])
    f_sr = os.path.join(g_srLocation, g_images[2][0], "64-512", g_images[2][1])
    imgs = [f_up, f_sr]
    for c in range(2):
        ax = axs[c]
        img = Image.open(imgs[c])
        ax.imshow(img)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_xlabel(c_titles[c])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.92, wspace=0.0, hspace=0.01)
    fig.suptitle("Subjective Comparison - Linear Interpolation vs SR")
    plt.savefig("../website/images/results/subjective_compare_1x2.png", bbox_inches='tight')
    plt.show()

def compare_checkerboard():
    for r in range(4):
        f_up = os.path.join(g_upscaleLocation, g_images[r][0], "64-512", g_images[r][1])
        f_sr = os.path.join(g_srLocation, g_images[r][0], "64-512", g_images[r][1])
        im_up = skimage.io.imread(f_up)
        im_sr = skimage.io.imread(f_sr)
        im_compR = skimage.util.compare_images(im_up[:,:,0], im_sr[:,:,0], method='checkerboard', n_tiles=(3,3))
        im_compG = skimage.util.compare_images(im_up[:,:,1], im_sr[:,:,1], method='checkerboard', n_tiles=(3,3))
        im_compB = skimage.util.compare_images(im_up[:,:,2], im_sr[:,:,2], method='checkerboard', n_tiles=(3,3))
        
        im_comp = np.dstack((im_compR, im_compG, im_compB))

        fig = plt.figure(figsize=[8, 8], dpi=150)
        axs = fig.subplots()
        axs.imshow(im_comp)
        axs.axvline(512 * 0.33)
        axs.axvline(512 * 0.66)
        axs.axhline(512 * 0.33)
        axs.axhline(512 * 0.66)
        plt.show()

def main():
    """main function"""    
    compare4x4()
    compare1x2()
    #compare_checkerboard()

if __name__ == '__main__':
    main()
