import torch
import torch.nn as nn   

class RECONSTRUCT(nn.Module):
    def __init__(self, num_class):
        super(RECONSTRUCT, self).__init__()
        self.num_class = num_class
        self.l1 = nn.Linear(17*num_class, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 784)
    def forward(self, pos, x):
        pos = pos.reshape([-1, 16*self.num_class])/100
        #print(pos.shape)
        #print(pos)
        x = torch.cat([pos,x], dim=1)
        #print(x.shape)
        #print(x)
        x = nn.ReLU()(self.l1(x))
        x = nn.ReLU()(self.l2(x))
        x = nn.Sigmoid()(self.l3(x))
        x = x.reshape([-1, 1,28,28])
        return x
