#
# Resizes image files for UW Madison CS766 Project, Mar 2020
# Eric Brandt, Asher Elmquist
#
#
import os
import cv2

# Filenames
f_downloadLocation = "downloads"
f_resizeLocation = "resized"
g_sizes = [2048, 1024, 512, 256, 128, 64]
g_startFromFullEachTime = False;


def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def downsample(img, sz):
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
    if img.shape[0] < g_sizes[0] or img.shape[1] < g_sizes[0] or not img.shape[2] == 3:
        raise Exception(f"image dimension less than {g_sizes[0]} too small ({img.shape})")

    # center crop to biggest size, square
    y = int((img.shape[0] - g_sizes[0]) / 2)
    x = int((img.shape[1] - g_sizes[0]) / 2)
    crop_img = img[y:y+g_sizes[0], x:x+g_sizes[0]]
    out_fname = os.path.join(outdir, str(g_sizes[0]), os.path.basename(fname))
    save_img(out_fname, crop_img)

    if g_startFromFullEachTime:
        down_img = crop_img.copy()
    else:
        down_img = crop_img

    for sz in g_sizes[1:]:
        down_img = downsample(down_img, sz)
        out_fname = os.path.join(outdir, str(sz), os.path.basename(fname))
        save_img(out_fname, down_img)
        if g_startFromFullEachTime:
            down_img = crop_img.copy()


def main():
    """main function"""

    if not os.path.exists(f_resizeLocation):
        os.makedirs(f_resizeLocation)

    # make a list of all the files to be processed.
    fils = []
    for root, dirs, files in os.walk(f_downloadLocation, topdown = False):
       for in_fname in files:
           if not os.path.basename(in_fname) == "_pending.jpg":
               fils.append(os.path.join(root, in_fname))
    print("Found {} files".format(len(fils)))

    cnt = 1
    start_cnt = 1
    errors = []
    for fil in fils:
        if cnt > start_cnt:
            outdir = os.path.dirname(fil);
            outdir = outdir.replace(f_downloadLocation, f_resizeLocation)
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
    print(f"{len(errors)} occurred.")
    for err in errors:
        print(f"Error {cnt}: ({err[0]}) {err[1]}")
        cnt = cnt + 1

if __name__ == '__main__':
    main()
