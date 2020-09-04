import torch
from torch import nn, optim
import numpy as np
import time 

class Generator(nn.Module):

    def __init__(self,classes,input_acc_size,input_nc=1,output_nc=1,ngf=64,n_blocks=4):
        super(Generator,self).__init__()

        self.classes = classes
        self.input_acc_size = input_acc_size
        
        # label emmbedding
        model = [nn.Embedding(classes,input_acc_size)]
        self.embed_model = nn.Sequential(*model)

        # concatenated 
        model = [nn.ReflectionPad1d(1),
                 nn.Conv1d(input_nc,ngf,kernel_size=8,padding=0,bias=True),
                 nn.InstanceNorm1d(ngf),
                 nn.ReLU(True)]

        n_downsampling=2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv1d(ngf*mult,ngf*mult*2,kernel_size = 3,stride=2,padding=1,bias=True),
                      nn.InstanceNorm1d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult)]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling -i)
            model += [nn.ConvTranspose1d(ngf * mult,int(ngf * mult /2),
                      kernel_size=4,stride = 2,padding=1,output_padding=1,bias=True),
                      nn.InstanceNorm1d(int(ngf * mult/2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad1d(3)]
        model += [nn.Conv1d(ngf,output_nc,kernel_size = 8,padding=1)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)

    def forward(self,z,labels):
        bache_size = len(z)
        embeded_labels = self.embed_model(labels).view(bache_size,1,self.input_acc_size)
        mul_tensor = torch.mul(z,embeded_labels)
        return self.model(mul_tensor)


class ResnetBlock(nn.Module):

    def __init__(self,dim):
        super(ResnetBlock,self).__init__()
        self.conv_block = self.build_conv_block(dim)
    
    def build_conv_block(self,dim):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad1d(1)]
        conv_block += [nn.Conv1d(dim,dim,kernel_size = 3,padding=0,bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)]
        conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad1d(1)]
        conv_block += [nn.Conv1d(dim,dim,kernel_size=3,padding=0,bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)]
        return nn.Sequential(*conv_block)

    def forward(self,x):
        out = x + self.conv_block(x)
        return out

if __name__ == "__main__":
    gyro_channel = 3
    classes = 10
    batches = 175
    input_size = 180
    gen = Generator(classes=classes,input_acc_size=input_size)

    s_time = time.time()
    for i in range(1500//batches+1):
        z = torch.FloatTensor(np.random.normal(0, 1, (batches, 1,input_size)))
        # print(z.shape)
        labels = torch.from_numpy(np.random.randint(0,classes-1,batches))

        output_wave = gen(z,labels)
        print(output_wave.shape)
    print(int((time.time()-s_time)*1000))
