import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


class Model_mnist(nn.Module):
    def __init__(self, gpu=False, **kwards):
        super(Model_mnist, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x
    def FeatureExtraction(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
    
class Model_cifar10(nn.Module):
    def __init__(self, gpu=False, num_class=10,**kwards):
        super(Model_cifar10, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(64*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_class)

        

    def FeatureExtraction(self, x):
       
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv12(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        
        return x
    def forward(self, x):
       
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv12(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

class Model_euroat(nn.Module):
    def __init__(self, gpu=False, num_class=10, **kwards):
        super(Model_euroat, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(64*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, num_class)

        

    def forward(self, x):
       
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv12(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x
    def FeatureExtraction(self, x):
       
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv12(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        return x
    
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

#8, 10, 0, 10
class Wide_ResNet(nn.Module):
    def __init__(self, depth=22, widen_factor=10, dropout_rate=0,gpu=True, num_class=100):
        super(Wide_ResNet, self).__init__()
        self.gpu=gpu
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_class)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.gpu==True:
            x=x.cuda()
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def FeatureExtraction(self, x):
        if self.gpu==True:
            x=x.cuda()
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        

        return out

    def loss(self, pred, label):
        if self.gpu==True:
            label=label.cuda()
        return F.cross_entropy(pred, label)