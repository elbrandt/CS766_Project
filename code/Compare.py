#
# Compare Image Files from 2 directory trees
# Eric Brandt, Asher Elmquist
#
#
import os
import cv2
import glob
import skimage
# from skimage import measure
import skimage.metrics as metrics
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Global settings
f_domain = "Building"
f_sourceLocation = "../testdata/resized/" + f_domain + "/512"
f_baselineLocation = "../testdata/upsampled/" + f_domain + "/64-512"
f_srLocation = "../testdata/sr_01/" + f_domain + "/64-512"
f_resultsLocation = "../testdata/sr_01/" + f_domain + "/512"

def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def compare_imgs(im1, im2, fil_out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore skimage's deprecation warinings.
        
        # structural similarity index
        ss = metrics.structural_similarity(im1,im2,multichannel=True)
        # [ssim, im] = measure.compare_ssim(im1, im2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True, full=True)
        # if fil_out:
        #     im8 = skimage.img_as_ubyte(np.clip(im, -1.0, 1.0))
        #     save_img(fil_out, im8)
# 
        # mean square error
        mse = metrics.mean_squared_error(im1, im2)
# 
        # normalized root mean squared error
        nrmse = metrics.normalized_root_mse(im1, im2)
# 
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


def main():
    """main function"""

    if not os.path.exists(f_resultsLocation):
        os.makedirs(f_resultsLocation)

    # make a list of all the files to be processed.
    fils = []
    errors = []
    for root, dirs, files in os.walk(f_sourceLocation, topdown = False):
       for fil_source in files:
            if not os.path.basename(fil_source) == "_pending.jpg":
                try:
                    fil_baseline = os.path.join(f_baselineLocation, os.path.basename(fil_source))
                    fil_sr = os.path.join(f_srLocation, os.path.basename(fil_source))
                    if os.path.exists(fil_baseline) and os.path.exists(fil_sr):
                        fils.append((os.path.join(f_sourceLocation, fil_source), fil_baseline, fil_sr))
                    else:
                        raise Exception("Could not find corresponding file")
                except Exception as ex:
                    # store the errors in a list, to make it easier to see how things finished
                    errors.append((fil_source, ex))
                    print(f" error: {ex}")

    print("Found {} files".format(len(fils)))

    stats = np.empty([len(fils), 8])
    cnt = 0
    for fil in fils:
        # find the corresponding file in source2Location
        try:
            fil_out = os.path.join(f_resultsLocation, os.path.basename(fil[0]))
            im_src = cv2.imread(fil[0])
            im_base = cv2.imread(fil[1])
            im_sr = cv2.imread(fil[2])
            metrics_base = compare_imgs(im_src, im_base, None)
            metrics_sr = compare_imgs(im_src, im_sr, None)
            stats[cnt,:] = metrics_base + metrics_sr
            # print("{}: {}".format(os.path.basename(fil[0]), metrics))
        except Exception as ex:
            # store the errors in a list, to make it easier to see how things finished
            errors.append((fil[0], ex))
            print(f" error: {ex}")
        cnt = cnt + 1

    plot_results(stats)

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
