import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
import PIL

from models import *
from loaders import *

batch_size      = 1
num_samples     = 77
image_shape     = (3,512,512)

#default datatype
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":

    dir_64 = "test_imgs/segmentable/64/"
    dir_512 = "test_imgs/segmentable/512/"
    dir_512_up = "test_imgs/upsampledDog/64-512/"

    data_JIT_loader = Img2ImgJITTestLoader(dir_64,dir_512,dir_512_up,paired_samples=True,num_samples=num_samples,file_type=".jpg")
    data_loader = torch.utils.data.DataLoader(data_JIT_loader, batch_size=batch_size,shuffle=True,num_workers=4)

    model = torch.load("Building_24k").cuda()
    model.eval()

    sr_ssim = np.zeros(len(data_JIT_loader))
    up_ssim = np.zeros(len(data_JIT_loader))

    sr_deep = np.zeros(len(data_JIT_loader))
    up_deep = np.zeros(len(data_JIT_loader))

    sr_fcn = np.zeros(len(data_JIT_loader))
    up_fcn = np.zeros(len(data_JIT_loader))

    fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=True)
    fcn_resnet101.eval()

    deeplabv3_resnet101 = models.segmentation.deeplabv3_resnet101(pretrained=True)
    deeplabv3_resnet101.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i,data in enumerate(data_loader, 0):
        low_res,high_res,upsampled = data
        img_inputs = low_res.cuda()
        # for u in range(1,4):
        #     u_f = 2**u
        #     img_inputs.append(torch.from_numpy(np.random.uniform(-1,1,(low_res.shape[0],1,low_res.shape[3]*u_f,low_res.shape[3]*u_f)).astype(np.float32)).to(device))

        # low_res = low_res.to(device)
        start = time.time()
        prediction = model(img_inputs)
        end = time.time()
        print("Inference in:",end-start,"seconds")

        low_res = low_res.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5
        prediction = prediction.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5
        upsampled = upsampled.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5
        high_res = high_res.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5



        sr_ssim[i] = ssim(prediction,high_res,multichannel=True)
        up_ssim[i] = ssim(upsampled,high_res,multichannel=True)

        prediction_in = preprocess(prediction).unsqueeze(0)
        upsampled_in = preprocess(upsampled).unsqueeze(0)
        high_res_in = preprocess(high_res).unsqueeze(0)


        deep_high_res = deeplabv3_resnet101.cpu()(high_res_in)['out'][0].argmax(0)
        deep_upsampled = deeplabv3_resnet101.cpu()(upsampled_in)['out'][0].argmax(0)
        deep_sr = deeplabv3_resnet101.cpu()(prediction_in)['out'][0].argmax(0)
        deep_error_up = np.count_nonzero(deep_high_res.numpy().astype(np.int)-deep_upsampled.numpy().astype(np.int))
        deep_error_sr = np.count_nonzero(deep_high_res.numpy().astype(np.int)-deep_sr.numpy().astype(np.int))
        sr_deep[i] = deep_error_sr
        up_deep[i] = deep_error_up

        fcn_high_res = fcn_resnet101.cpu()(high_res_in)['out'][0].argmax(0)
        fcn_upsampled = fcn_resnet101.cpu()(upsampled_in)['out'][0].argmax(0)
        fcn_sr = fcn_resnet101.cpu()(prediction_in)['out'][0].argmax(0)
        fcn_error_up = np.count_nonzero(fcn_high_res.numpy().astype(np.int)-fcn_upsampled.numpy().astype(np.int))
        fcn_error_sr = np.count_nonzero(fcn_high_res.numpy().astype(np.int)-fcn_sr.numpy().astype(np.int))
        sr_fcn[i] = fcn_error_sr
        up_fcn[i] = fcn_error_up

        print("ssim up:",up_ssim[i])
        print("ssim sr:",sr_ssim[i])

        print("deep error up:",deep_error_up)
        print("deep error sr:",deep_error_sr)

        print("fcn error up:",fcn_error_up)
        print("fcn error sr:",fcn_error_sr)

        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([c for c in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        fcn_high_res = PIL.Image.fromarray(fcn_high_res.byte().cpu().numpy())
        fcn_high_res.putpalette(colors)

        deep_high_res = PIL.Image.fromarray(deep_high_res.byte().cpu().numpy())
        deep_high_res.putpalette(colors)

        fcn_upsampled = PIL.Image.fromarray(fcn_upsampled.byte().cpu().numpy())
        fcn_upsampled.putpalette(colors)

        deep_upsampled = PIL.Image.fromarray(deep_upsampled.byte().cpu().numpy())
        deep_upsampled.putpalette(colors)

        fcn_sr = PIL.Image.fromarray(fcn_sr.byte().cpu().numpy())
        fcn_sr.putpalette(colors)

        deep_sr = PIL.Image.fromarray(deep_sr.byte().cpu().numpy())
        deep_sr.putpalette(colors)

        f, ax = plt.subplots(3,3,figsize=(12,12))
        ax[0,0].imshow(high_res)
        ax[0,0].set_title("Ground Truth Img")
        ax[1,0].imshow(fcn_high_res)
        ax[1,0].set_title("FCN on GT")
        ax[2,0].imshow(deep_high_res)
        ax[2,0].set_title("Deeplab on GT")
        ax[0,1].imshow(upsampled)
        ax[0,1].set_title("Upsampled Img")
        ax[1,1].imshow(fcn_upsampled)
        ax[1,1].set_title("FCN on Upsampled")
        ax[2,1].imshow(deep_upsampled)
        ax[2,1].set_title("Deeplab on Upsampled")
        ax[0,2].imshow(prediction)
        ax[0,2].set_title("SR Img")
        ax[1,2].imshow(fcn_sr)
        ax[1,2].set_title("FCN on SR")
        ax[2,2].imshow(deep_sr)
        ax[2,2].set_title("Deeplab on SR")

        # plt.savefig("results/seg/seg_"+str(i)+".png",dpi=300)
        # plt.show()


    print("SR Mean SSIM:",np.mean(sr_ssim))
    print("Up Mean SSIM",np.mean(up_ssim))

    print("SR Mean deep Error:",np.mean(sr_deep) / (512*512))
    print("Up Mean deep Error:",np.mean(up_deep) / (512*512))

    print("SR Mean fcn Error:",np.mean(sr_fcn) / (512*512))
    print("Up Mean fcn Error:",np.mean(up_fcn) / (512*512))
