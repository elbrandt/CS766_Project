#
# Compare Image Files from 2 directory trees
# Eric Brandt, Asher Elmquist
#
#
import os
import matplotlib.pyplot as plt
from PIL import Image

# original files are in ../testdata/resized/[domain]/512"
g_origLocation = "../testdata/resized/"
g_upscaleLocation = "../testdata/upsampled/"
g_srLocation = "../testdata/sr/Building_1094"

g_images = [
    ( "Building", "000d6781b9abfae8.jpg" ),
    ( "Dog", "8a19c3b7d2007355.jpg" ),
    ( "Flower", "2d27fecf7183a2d7.jpg" ),
    ( "Food", "2d4a38d39b4db37d.jpg" )
]

def main():
    """main function"""    

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
    plt.savefig("../website/images/subjective_compare_4x4.png", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
