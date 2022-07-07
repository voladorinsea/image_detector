from turtle import Turtle, forward
from torch import conv2d, max_pool2d, tensor
import torch
import torch.nn as nn

'''those algorithm are used to detect human. 
   As a result, there are only one class as default'''
class Yolov1(nn.Module):
    def __init__(self, num_class = 1) -> None:
        super(Yolov1, self).__init__()
        self.feature = nn.Sequential(           # input[3, 448, 448]
            nn.Conv2d(3, 64, 7, 2, 3),          # output[64, 224, 224]
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # output[64, 112, 112]
            nn.Conv2d(64, 192, 3, 1, 1),        # output[192, 112, 112  
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # output[192, 56, 56]
            nn.Conv2d(192, 128, 1, 1, 0),       # output[128, 56, 56]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),       # output[256, 56, 56]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),       # output[256, 56, 56]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),       # output[512, 56, 56]
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # output[512, 28, 28]
            nn.Conv2d(512, 256, 1, 1, 0),       # output[256, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),       # output[512, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, 1, 1, 0),       # output[256, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),       # output[512, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, 1, 1, 0),       # output[256, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),       # output[512, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, 1, 1, 0),       # output[256, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),       # output[512, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),       # output[512, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1),      # output[1024, 28, 28]
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),                 # output[1024, 14, 14]
            nn.Conv2d(1024, 512, 1, 1, 0),      # output[512, 14, 14]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1),      # output[1024, 14, 14]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, 1, 0),      # output[512, 14, 14]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1),      # output[1024, 14, 14]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),     # output[1024, 14, 14]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 2, 1),     # output[1024, 7, 7]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),     # output[1024, 7, 7]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),     # output[1024, 7, 7]
            nn.LeakyReLU(inplace=True),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(7*7*1024, 4096),         # output[4096]
            nn.LeakyReLU(inplace=True),
        )
        self.linear2 = nn.Linear(4096, 7*7*num_class)    # output[7*7*30]
        self.num_class = num_class
    def forward(self, x):
        
        x = self.feature(x)
        x = x.view(-1, 7*7*1024)
        x = self.linear1(x)
        x = self.linear2(x)
        
        return x
class yolov3(nn.Module):
    def __init__(self, num_class, init_param : bool = False) -> None:
        super(yolov3, self).__init__()
        
if __name__ == '__main__':
    x = torch.ones([3, 448, 448])
    x = x.unsqueeze(0)
    print(x.shape)
    model = Yolov1(30)
    x = model(x)
    print(x.shape)
    print(model.modules)