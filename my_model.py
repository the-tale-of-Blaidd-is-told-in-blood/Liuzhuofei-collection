import torch
from torch import nn
from torch.nn import Flatten


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.struct1=nn.Sequential(
            nn.Conv2d(3,32,5,1,2,1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2,1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2,1),
            nn.MaxPool2d(2),

        )
        self.struct2=Flatten()
        self.struct3=nn.Sequential(
            nn.Linear(1024,64,True),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.struct1(x)
        x=self.struct2(x)
        x=self.struct3(x)
        return x

# x=torch.ones(3,32,32)
# model1=Mymodel()
# y=model1(x)
# print(y.shape)