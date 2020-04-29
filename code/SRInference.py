#
# Upsamples image files for UW Madison CS766 Project, Mar 2020
# Eric Brandt, Asher Elmquist
#
#
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from models import *
from models import *
from PIL import Image

# Global settings
#f_sourceLocation = "/home/amelmquist/datasets/sr"
#f_resizeLocation = "/home/amelmquist/datasets/sr/upsampled"
f_sourceLocation = "../testdata/resized"
f_model = "Food"
f_inferenceLocation = "../testdata/sr_" + f_model
g_startSize = 64
g_endSize = 512

g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
image_shape = (4,480,640)
#g_model = SRNet(image_shape=image_shape,model_name="Building",device=g_device,continue_from_save=False)
g_model = SRNet(image_shape=image_shape,device=g_device,continue_from_save=False)
g_model.load(f_model)

def ensure_dir_exists(fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def inference(img):
    img = (np.array(img).astype(np.float32) / 127.5) - 1
    img = img.transpose((2,0,1))
    t = np.expand_dims(img, 0)
    img_inputs = torch.from_numpy(t).to(g_device)
    prediction = g_model.test(img_inputs)
    out_img = prediction.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]
    out_img = (np.round(out_img * 127.5 + 127.5)).astype(np.uint8)
    
    return out_img

def save_img(fname, img):
    ensure_dir_exists(fname)
    #cv2.imwrite(fname, img)
    img = Image.fromarray(img)
    img.save(fname)
    

def inference_file(fname, outdir):
    img = Image.open(fname)
    
    if img.size[0] != g_startSize or img.size[1] != g_startSize or img.layers != 3:
        raise Exception(f"image dimension {img.shape[0]}x{img.shape[1]} is not {g_startSize}x{g_startSize}")

    up_img = inference(img)

    out_fname = os.path.join(outdir, os.path.basename(fname))
    save_img(out_fname, up_img)


def main():
    """main function"""

    if not os.path.exists(f_inferenceLocation):
        os.makedirs(f_inferenceLocation)

    # make a list of all the files to be processed.
    sourceLoc = "{}/**/{}/*.jpg".format(f_sourceLocation, g_startSize)
    fils = glob.glob(sourceLoc)
    print("Found {} files".format(len(fils)))

    cnt = 1
    errors = []
    for fil in fils:
        outdir = os.path.dirname(fil)
        outdir = outdir.replace(f_sourceLocation, f_inferenceLocation)
        outdir = outdir.replace(str(g_startSize), "{}-{}".format(g_startSize, g_endSize))
        print("Processing ({}/{}) {}=>{}".format(cnt, len(fils), fil, outdir))
        try:
            inference_file(fil, outdir)
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
