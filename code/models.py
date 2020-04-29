import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import models, transforms


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.net = models.vgg19(pretrained=True).features
        # self.net._modules['4'] = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)
        # self.net._modules['9'] = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)
        # self.net._modules['18'] = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)
        # self.net._modules['27'] = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)
        # self.net._modules['36'] = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)

        for p in self.net.parameters():
            p.requires_grad = False

        self.net.eval()
        self.net.cuda()

        self.rmean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).cuda()
        self.rstd = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).cuda()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

        # self.feature_layers = [8,17,26,35]
        self.feature_layers = [8,17,26,35]

    def forward(self, x):
        x = ((x*self.rstd + self.rmean) - self.mean) / self.std

        outputs = []
        for n, l in enumerate(self.net):
            x = l(x)
            if n in self.feature_layers:
                outputs.append(x)
                if len(outputs) >= len(self.feature_layers):
                    break
        return outputs


########################
class SRNet():
    def __init__(self,image_shape,model_name="model",device=torch.device("cpu"),continue_from_save=False):
        self.device = device
        self.continue_from_save = continue_from_save
        self.c=image_shape[0]
        self.h=image_shape[1]
        self.w=image_shape[2]
    
        self.name = model_name

        self.up = 1
        self.max_up = 3 #how far will we want to go? 2**n = 2048 ==> n=5

        self.pixel_loss_weight = 10
        self.perceptual_loss_weight = [1,.7,.5,0] #[1,.1,.01,.1]

        self.net = SRResNet(f=64,up=self.up,max_up=self.max_up)

        #load weights from file if we are to continue from previous training
        if self.continue_from_save:
            self.load("saved_models/"+self.name)

        #put the networks on the gpu
        self.net.to(device)

    def grow(self):
        self.net.grow_net()
        self.up += 1

    def train(self,data_loader,num_epochs=2,batch_size=1,num_samples=-1,lr=1e-4,print_interval=10,save_interval=1,img_progress_interval=-1):
        self.data_loader = data_loader
        self.batch_size = batch_size

        self.num_samples = num_samples if num_samples>0 else self.data_loader.__len__()*self.batch_size
        print("Num samples:",self.num_samples)
        self.lr = lr
        self.print_interval = print_interval
        self.save_interval = save_interval
        self.num_epochs = num_epochs

        #initialize optimizers
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        # self.loss_function = nn.L1Loss()
        # self.loss_function = nn.MSELoss()
        # self.loss_function = nn.L1Loss()
        self.loss_function = nn.MSELoss()
        self.loss_net = PerceptualLoss()
        self.loss_net.eval()

        #training loop
        t0 = time.time()
        for e in range(self.num_epochs):

            #zero out our loss trackers
            loss_sum = np.zeros((2))
            num_steps = 0 #counter for discriminator updates
            for i,data in enumerate(self.data_loader, 0):
                num_steps += 1
                #grab some real samples
                img_low,img_high = data

                # img_inputs = [img_low.to(self.device)]
                img_inputs = img_low.to(self.device)
                # for u in range(1,self.up+1):
                #     u_f = 2**u
                #     img_inputs.append(torch.from_numpy(np.random.uniform(-1,1,(img_low.shape[0],1,img_low.shape[3]*u_f,img_low.shape[3]*u_f)).astype(np.float32)).to(self.device))

                    # print("Len:",len(img_inputs))
                # print(img_inputs)
                # img_low = torch.cat([img_low,img_noise],1)
                # img_low = img_low.to(self.device)
                img_high = img_high.to(self.device)

                #zero the gradients
                self.opt.zero_grad()
                #make prediction
                pred_high = self.net(img_inputs)
                #propagate loss

                loss_pix = self.pixel_loss_weight*self.loss_function(pred_high,img_high)
                pred_features = self.loss_net(pred_high)
                target_features = self.loss_net(img_high)
                # loss_f1 = self.perceptual_loss_weight[0]*torch.mean((pred_features[0] -  target_features[0])**2)
                # loss_f2 = self.perceptual_loss_weight[1]*torch.mean((pred_features[1] -  target_features[1])**2)
                # loss_f3 = self.perceptual_loss_weight[2]*torch.mean((pred_features[2] -  target_features[2])**2)
                # loss_f4 = self.perceptual_loss_weight[3]*torch.mean((pred_features[3] -  target_features[3])**2)

                loss_tex = 0
                gram_pred = [gram_matrix(y) for y in pred_features]
                gram_target = [gram_matrix(y) for y in target_features]
                for m in range(0, len(gram_pred)):
                    loss_tex += 1000*torch.mean((gram_pred[m] - gram_target[m])**2)
                # text_loss = sum(text_loss)

                # loss = loss_pix+loss_f1+loss_f2+loss_f3+loss_f4
                # loss = loss_pix+loss_tex
                loss = 0
                if(e<10):
                    loss = loss_pix
                else:
                    loss = loss_pix + loss_tex

                loss.backward()
                # loss_sum += np.array([loss_pix.item(),loss_f1.item(),loss_f2.item(),loss_f3.item(),loss_f4.item()])
                # print(loss_pix.item())
                # print(text_loss.item())
                loss_sum += np.array([loss_pix.item(),loss_tex.item()])
                #step the optimizer
                self.opt.step()

                #print when needed
                if(i%self.print_interval == self.print_interval-1):
                    print('[%d/%d],[%d/%d]'%#', C_ABA: %.6f, C_BAB: %.6f' %
                        (e + 1, self.num_epochs,
                        i + 1, self.num_samples/self.batch_size),  "Losses:",
                        loss_sum / print_interval#,
                        )

                    loss_sum = np.zeros((2))
                    num_steps = 0

                #save images if needed
                if (img_progress_interval>0 and i % img_progress_interval == img_progress_interval-1):
                    img_low,img_high = next(iter(data_loader))
                    #grab some real samples
                    img_low,img_high = data
                    # img_inputs = [img_low.to(self.device)]
                    img_inputs = img_low.to(self.device)
                    # for u in range(1,self.up+1):
                    #     u_f = 2**u
                    #     img_inputs.append(torch.from_numpy(np.random.uniform(-1,1,(img_low.shape[0],1,img_low.shape[3]*u_f,img_low.shape[3]*u_f)).astype(np.float32)).to(self.device))

                    # img_noise = torch.from_numpy(np.random.uniform(-1,1,img_low.shape).astype(np.float32))
                    # img_low_cat = torch.cat([img_low,img_noise],1)
                    # img_low_cat = img_low_cat.to(self.device)
                    img_high = img_high.to(self.device)

                    pred_img_high = self.test(img_inputs)

                    f, ax = plt.subplots(1,3)
                    ax[0].imshow(img_low.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
                    ax[0].set_title('Low Res')

                    ax[1].imshow(pred_img_high.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
                    ax[1].set_title('Predicted High Res')

                    ax[2].imshow(img_high.detach().cpu().numpy().transpose((0, 2, 3, 1))[0,:,:,:]*.5+.5)
                    ax[2].set_title('Real High Res')

                    plt.savefig("progress/"+self.name+"_"+str(self.up)+"_"+str(e)+"_"+str(i)+".png")
                    plt.close(f)

            #save when needed
            if (e % self.save_interval == self.save_interval-1):
                self.save("saved_models/"+self.name)
                self.export()

    def test(self,inputs):
        sr_img = self.net(inputs)
        return sr_img

    def load(self,path):
        self.net = torch.load(path)
        # self.net.load("saved_models/net")

    def save(self,path):
        torch.save(self.net, path)

        # if torch.cuda.device_count() > 1:
        #     self.net.module.save("saved_models/net")
        # else:
        #     self.net.save("saved_models/net")

    def export(self):
        pass



class SRResNet(nn.Module):
    def __init__(self,f=8,up=1,max_up=5):    #upscale in 2**up
        super(SRResNet,self).__init__()

        self.f = f
        self.up = up

        #activations
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        #layers
        #expanding input block for image
        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=f,kernel_size=3,stride=1,padding=1,padding_mode="reflection")
        self.in1_1 = nn.InstanceNorm2d(f)
        # self.conv1_2 = nn.Conv2d(in_channels=4*f,out_channels=8*f,kernel_size=3,stride=1,padding=1)
        # self.in1_2 = nn.InstanceNorm2d(8*f)
        # self.conv1_3 = nn.Conv2d(in_channels=8*f,out_channels=8*f,kernel_size=3,stride=1,padding=1)
        # self.in1_3 = nn.InstanceNorm2d(8*f)

        #upsampling
        self.num_blocks = 4
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.conv2_1 = nn.Conv2d(in_channels=f,out_channels=f,kernel_size=3,stride=1,padding=1,padding_mode="reflection")
            self.in2_1 = nn.InstanceNorm2d(8*f)
            self.conv2_2 = nn.Conv2d(in_channels=f,out_channels=f,kernel_size=3,stride=1,padding=1,padding_mode="reflection")
            self.in2_2 = nn.InstanceNorm2d(8*f)
            # self.conv2_3 = nn.Conv2d(in_channels=16*f,out_channels=16*f,kernel_size=3,stride=1,padding=1)
            # self.in2_3 = nn.InstanceNorm2d(16*f)
            # self.conv2_4 = nn.Conv2d(in_channels=16*f,out_channels=16*f,kernel_size=3,stride=1,padding=1)
            # self.in2_4 = nn.InstanceNorm2d(16*f)
            # self.conv2_5 = nn.Conv2d(in_channels=16*f,out_channels=32*f,kernel_size=3,stride=1,padding=1)
            # self.in2_5 = nn.InstanceNorm2d(32*f)
            # self.shuffle = nn.PixelShuffle(2)
            self.blocks.append(nn.ModuleList([self.conv2_1,self.in2_1,self.conv2_2,self.in2_2]))#,self.conv2_3,self.in2_3,self.conv2_4,self.in2_4,self.conv2_5,self.in2_5,self.shuffle]))

        self.ups = nn.ModuleList()
        for i in range(max_up):
            self.conv_up = nn.Conv2d(in_channels=f,out_channels=4*f,kernel_size=3,stride=1,padding=1,padding_mode="reflection")
            self.shuffle = nn.PixelShuffle(2)
            self.ups.append(nn.ModuleList([self.conv_up,self.shuffle]))#,self.conv2_3,self.in2_3,self.conv2_4,self.in2_4,self.conv2_5,self.in2_5,self.shuffle]))


        #finishing block for CNN
        # self.conv_out_1 = nn.Conv2d(in_channels=8*f,out_channels=8*f,kernel_size=5,stride=1,padding=2)
        # self.in_out_1 = nn.InstanceNorm2d(8*f)
        # self.conv_out_2 = nn.Conv2d(in_channels=8*f,out_channels=8*f,kernel_size=3,stride=1,padding=1)
        # self.in_out_2 = nn.InstanceNorm2d(8*f)
        # self.conv_out_3 = nn.Conv2d(in_channels=8*f,out_channels=8*f,kernel_size=3,stride=1,padding=1)
        # self.in_out_3 = nn.InstanceNorm2d(8*f)
        self.conv_out_4 = nn.Conv2d(in_channels=f,out_channels=3,kernel_size=3,stride=1,padding=1,padding_mode="reflection")

        #initialize weights
        self.init_weights()

    def forward(self,x):
        # inputs = x
        # x = inputs[0]
        x = self.relu(self.conv1_1(x))
        # x = self.relu(self.in1_2(self.conv1_2(x)))
        # x = self.relu(self.in1_3(self.conv1_3(x)))
        for i in range(self.num_blocks):
            r = self.relu(self.blocks[i][1](self.blocks[i][0](x)))
            r = self.relu(self.blocks[i][3](self.blocks[i][2](r)))
            x = r + x

        for i in range(self.up):
            #one layer to handle noise concatenation
            x = self.relu(self.ups[i][0](x))
            x = self.ups[i][1](x)
            #ResNet Block
            # x_save = x
            # x = self.relu(self.blocks[i][3](self.blocks[i][2](x)))
            # x = self.relu(self.blocks[i][5](self.blocks[i][4](x)))
            # x = self.relu(self.blocks[i][7](self.blocks[i][6](x)))
            # x = torch.add(x,x_save)

            #upscaling layer
            # x = self.relu(self.blocks[i][9](self.blocks[i][8](x)))


            #noise introcuction for details at next higher resolution
            # x = torch.cat([x,inputs[i+1]],1)


        # x = self.relu(self.in_out_1(self.conv_out_1(x)))
        # x = self.relu(self.in_out_2(self.conv_out_2(x)))
        # x = self.relu(self.in_out_3(self.conv_out_3(x)))
        x = self.tanh(self.conv_out_4(x))

        return x

    def grow_net(self):
        self.prev_up = self.up
        self.up += 1

        for i in range(self.prev_up,self.up):
            # init.orthogonal_((self.ups[i][0]).weight, init.calculate_gain('relu'))
            init.xavier_uniform_(self.ups[i][0].weight, gain=np.sqrt(2))
            # init.orthogonal_(self.blocks[i][2].weight, init.calculate_gain('relu'))
            # init.orthogonal_(self.blocks[i][4].weight, init.calculate_gain('relu'))

    def init_weights(self):
        # init.orthogonal_(self.conv1_1.weight, init.calculate_gain('relu'))
        init.xavier_uniform_(self.conv1_1.weight, gain=np.sqrt(2))
        # init.orthogonal_(self.conv1_2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv1_3.weight, init.calculate_gain('relu'))

        for i in range(self.num_blocks):
            # init.orthogonal_((self.blocks[i][0]).weight, init.calculate_gain('relu'))
            init.xavier_uniform_(self.blocks[i][0].weight, gain=np.sqrt(2))
            # init.orthogonal_(self.blocks[i][2].weight, init.calculate_gain('relu'))
            init.xavier_uniform_(self.blocks[i][2].weight, gain=np.sqrt(2))
            # init.orthogonal_(self.blocks[i][4].weight, init.calculate_gain('relu'))
            # init.orthogonal_(self.blocks[i][6].weight, init.calculate_gain('relu'))
        for i in range(self.up):
            # init.orthogonal_((self.ups[i][0]).weight, init.calculate_gain('relu'))
            init.xavier_uniform_(self.ups[i][0].weight, gain=np.sqrt(2))
            # init.orthogonal_(self.ups[i][2].weight, init.calculate_gain('relu'))

        # init.orthogonal_(self.conv2_1.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv2_2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv2_3.weight, init.calculate_gain('relu'))

        # init.orthogonal_(self.conv_out_1.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv_out_2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv_out_3.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv_out_4.weight, init.calculate_gain('tanh'))
        init.xavier_uniform_(self.conv_out_4.weight, gain=np.sqrt(2))
        # init.orthogonal_(self.conv9_4.weight, init.calculate_gain('tanh'))

    def save(self,path):
        torch.save(self.state_dict(),path)

    def load(self,path):
        self.load_state_dict(torch.load(path))
