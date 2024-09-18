# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class _Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        kernel_size = 3
        padding = (kernel_size - 1)//2 if kernel_size % 2 == 1 else kernel_size // 2

        self.main = nn.Sequential(
                # size
                nn.Conv2d(1, 4, kernel_size=kernel_size, stride=1, padding=padding),  
                #nn.BatchNorm2d(8),
                nn.ReLU(),

                nn.Conv2d(4, 8, kernel_size=kernel_size, stride=1, padding=padding), 
                #nn.BatchNorm2d(12),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),

                nn.Conv2d(8, 16, kernel_size=kernel_size, stride=1, padding=padding), # size
                #nn.BatchNorm2d(16),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
            )

        # self.linear = nn.Linear((28 // 4) * ((56 // 4))  * 16 , 64)
        # self.output_function = nn.ReLU() # or using LeakyReLU, Tanh, Sigmoid, ...etc
        
        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, mean = 0.0 , std=0.01)
                init.zeros_(m.bias)


    def forward(self, x):

        # np.save("./data1.npy", x.detach().cpu().numpy())
        # y1 = self.main[:2](x)
        # y2 = self.main[:4](x)
        # np.save("./y1.npy", y1.detach().cpu().numpy())
        # np.save("./y2.npy", y2.detach().cpu().numpy())
        
        x = self.main(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.linear(x)
        # x = self.output_function(x)

        return x



class _Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        # self.linear = nn.Linear(64, (28 // 4) * ((56 // 4))  * 16)
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.relu = nn.ReLU()

        kernel_size = 3
        padding = (kernel_size - 1)//2 if kernel_size % 2 == 1 else kernel_size // 2

        self.main = nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=kernel_size, stride=1, padding=padding), # size
                # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                #nn.BatchNorm2d(12),
                nn.ReLU(),
                # nn.ZeroPad2d((0,0,0,1)),

                nn.Conv2d(8, 4, kernel_size=kernel_size, stride=1, padding=padding), # size
                # nn.ConvTranspose2d(12, 8, kernel_size=4, stride=2, padding=1),
                #nn.BatchNorm2d(8),
                nn.ReLU(),

                nn.Conv2d(4, 1, kernel_size=kernel_size, stride=1, padding=padding), # size
                # nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(1),
                nn.Sigmoid()
            )        

        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, mean = 0.0 , std=0.01)
                init.zeros_(m.bias)


    def forward(self, x):

        # x = self.linear(x)
        # x = self.relu(x)
        # x = x.view(x.size(0), -1, 28//4, 56//4)
        x = self.main(x)
        return x




class Downstream_Task_Model(nn.Module):

    def __init__(self):

        super().__init__()

        # kernel_size = 3
        # padding = kernel_size // 2
        # self.main = nn.Sequential(
        #         nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1),
        #         nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        #         nn.Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        #         nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1),
        #         nn.AdaptiveAvgPool2d((1,1)),
        #     )        
        # self.linear = nn.Linear(64 , 5)

        self.main = Wide_ResNet(input_channels=3, num_classes=5, dropout_rate = 0.25)
        self.softmax = nn.Softmax(dim = -1)

        # self.initialize()



    def initialize(self):
        c = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, mean = 0.0 , std=0.01)
                init.zeros_(m.bias)
                c += 1
        print(f"model initialization, #modules: {c}")

    def __forward(self, x):

        x = self.main(x)
        # x = torch.squeeze(x)
        # x = self.linear(x)
        return x

    def posterior_predict(self, x = None, logit = None):
        if logit is None:
            assert x is not None
            logit = self.__forward(x)
        prediction = self.softmax(logit)
        return prediction
    
    def energy_score(self, x = None, logit = None):
        if logit is None:
            assert x is not None
            logit = self.__forward(x)
        energy = -1 * torch.logsumexp(logit, dim = -1)
        return energy


    def forward(self, x):    
        logit = self.__forward(x)

        # prediction = self.posterior_predict(logit = logit)
        prediction = logit  # By default, we will use built-in torch.CrossEntropy Class,
                            # which has already integrate the logSoftmax and NLL-Loss
                            # hence, here we don't need to manually apply the softmax function
                            # when only in the inference stage and you want to get a probabilistic prediction
                            # you can call self.posterior_predicit(*) to get a softmax output
        energy_score = self.energy_score(logit = logit)
        return prediction, energy_score





def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class Wide_ResNet(nn.Module):
    def __init__(self, depth = 28, widen_factor = 10, num_classes=10, input_channels=3,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('[!] Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, vx=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



if __name__ == "__main__":

    # encoder = _Encoder()
    # decoder = _Decoder()

    # # unit-test
    # a_batch_of_images = torch.randn(7 , 1 , 175 , 215)
    # code = encoder(a_batch_of_images)
    # decoded_image = decoder(code)
    # print(code.shape, decoded_image.shape)

    # # Gradient unit-test
    # loss = torch.mean(
    #             torch.sum(
    #                 torch.square(decoded_image - a_batch_of_images) , dim = (1,2,3)
    #             )
    #         )
    # loss.backward()
    # print(loss.item())


    # downstream_task_model unit-test
    task_model = Downstream_Task_Model()
    a_batch_of_images = torch.randn(17, 3, 32, 32)
    
    prediction = task_model.posterior_predict(a_batch_of_images)
    print(prediction.shape)

    energy_score = task_model.energy_score(a_batch_of_images)
    print(energy_score.shape)