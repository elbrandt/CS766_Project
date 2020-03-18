#
# Upsamples image files for UW Madison CS766 Project, Mar 2020
# Eric Brandt, Asher Elmquist
#
#
import os
import cv2
import glob

# Global settings
f_sourceLocation = "resized"
f_resizeLocation = "upsampled"
g_startSize = 64
g_endSize = 512
g_oneShotUpsize = True; # true=? start->end in one rescale, false=>rescale x2 repeatedly


def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def upsample(img, sz):
    out_img = cv2.resize(img, (sz, sz))
    return out_img

def save_img(fname, img):
    ensure_dir_exists(fname)
    if not os.path.exists(fname):
        cv2.imwrite(fname, img)
    else:
        print(f" skipping writing {fname}. File already exists")

def resize_file(fname, outdir):
    img = cv2.imread(fname)
    if img.shape[0] != g_startSize or img.shape[1] != g_startSize or img.shape[2] != 3:
        raise Exception(f"image dimension {img.shape[0]}x{img.shape[1]} is not {g_startSize}x{g_startSize}")

    if g_oneShotUpsize:
        up_img = upsample(img, g_endSize)
    else:
        sz = g_startSize
        up_img = img;
        while sz < g_endSize:
            sz = sz * 2
            up_img = upsample(up_img, sz)

    out_fname = os.path.join(outdir, os.path.basename(fname))
    save_img(out_fname, up_img)

        
def main():
    """main function"""

    if not os.path.exists(f_resizeLocation):
        os.makedirs(f_resizeLocation)

    # make a list of all the files to be processed.
    sourceLoc = "{}/**/{}/*.jpg".format(f_sourceLocation, g_startSize)
    fils = glob.glob(sourceLoc)
    print("Found {} files".format(len(fils)))

    cnt = 1
    errors = []
    for fil in fils:
        outdir = os.path.dirname(fil);
        outdir = outdir.replace(f_sourceLocation, f_resizeLocation)
        outdir = outdir.replace(str(g_startSize), "{}-{}".format(g_startSize, g_endSize))
        print("Processing ({}/{}) {}=>{}".format(cnt, len(fils), fil, outdir))
        try:
            resize_file(fil, outdir)
        except Exception as ex:
            # store the errors in a list, to make it easier to see how things finished
            errors.append((fil, ex))
            print(f" error: {ex}") 
        cnt = cnt + 1

    # print out the list of errors at the end
    cnt = 1
    print(f"{len(errors)} errors occurred.")
    for err in errors:
        print(f"Error {cnt}: ({err[0]}) {err[1]}")
        cnt = cnt + 1

if __name__ == '__main__':
    main()
