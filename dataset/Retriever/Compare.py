#
# Compare Image Files from 2 directory trees
# Eric Brandt, Asher Elmquist
#
#
import os
import cv2
import glob
import skimage
from skimage import measure
import numpy as np
import warnings

# Global settings
f_source1Location = "resized/Food/512"
f_source2Location = "upsampled/Food/64-512"
f_resultsLocation = "comparison/Food/512"


def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def upsample(img, sz):
    out_img = cv2.resize(img, (sz, sz))
    return out_img

def save_img(fname, img):
    ensure_dir_exists(fname)
    cv2.imwrite(fname, img)

def compare_imgs(im1, im2, fil_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore skimage's deprecation warinings.
        
        # structural similarity index
        [ssim, im] = measure.compare_ssim(im1, im2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True, full=True)
        im8 = skimage.img_as_ubyte(np.clip(im, -1.0, 1.0))
        save_img(fil_out, im8)

        # mean square error
        mse = measure.compare_mse(im1, im2)

        # normalized root mean squared error
        nrmse = measure.compare_nrmse(im1, im2)

        # peak signal-to-noise ratio
        psnr = measure.compare_psnr(im1, im2)

        return [ssim, mse, nrmse, psnr]

def main():
    """main function"""

    if not os.path.exists(f_resultsLocation):
        os.makedirs(f_resultsLocation)

    # make a list of all the files to be processed.
    fils = []
    errors = []
    for root, dirs, files in os.walk(f_source1Location, topdown = False):
       for fil1 in files:
            if not os.path.basename(fil1) == "_pending.jpg":
                try:
                    fil2 = os.path.join(f_source2Location, os.path.basename(fil1))
                    if os.path.exists(fil2):
                        fils.append((os.path.join(f_source1Location, fil1), fil2))
                    else:
                        raise Exception("Could not find corresponding file")
                except Exception as ex:
                    # store the errors in a list, to make it easier to see how things finished
                    errors.append((fil1, ex))
                    print(f" error: {ex}")

    print("Found {} files".format(len(fils)))

    stats = np.empty([len(fils), 4])
    cnt = 0
    for fil in fils:
        # find the corresponding file in source2Location
        try:
            fil_out = os.path.join(f_resultsLocation, os.path.basename(fil[0]))
            im1 = cv2.imread(fil[0])
            im2 = cv2.imread(fil[1])
            metrics = compare_imgs(im1, im2, fil_out)
            stats[cnt,:] = metrics
            # print("{}: {}".format(os.path.basename(fil[0]), metrics))
        except Exception as ex:
            # store the errors in a list, to make it easier to see how things finished
            errors.append((fil[0], ex))
            print(f" error: {ex}")
        cnt = cnt + 1

    # print out the list of errors at the end
    cnt = 1
    print(f"{len(errors)} errors occurred.")
    for err in errors:
        print(f"Error {cnt}: ({err[0]}) {err[1]}")
        cnt = cnt + 1

    means = np.mean(stats, axis=0)
    stddevs = np.std(stats, axis=0)
    print(means)
    print(stddevs)

    np.savetxt(os.path.join(f_resultsLocation, 'stats.csv'), stats, delimiter=',')

if __name__ == '__main__':
    main()
